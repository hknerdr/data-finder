from flask import Flask, render_template, request, jsonify, send_file
import requests
from bs4 import BeautifulSoup
import threading
import time
import re
import phonenumbers
import logging
from urllib.parse import urlparse, quote
import datetime
from datetime import datetime, timedelta
import nltk
from nltk.tokenize import sent_tokenize
import json
import io
import csv
import os
from fake_useragent import UserAgent
import itertools
from tenacity import retry, stop_after_attempt, wait_exponential
import random
import socket
import socks
from requests.exceptions import RequestException, ProxyError
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration classes
class Config:
    RETRY_ATTEMPTS = 3
    REQUEST_TIMEOUT = 30
    RATE_LIMIT_DELAY = 2
    MAX_RESULTS_PER_QUERY = 20
    USER_AGENT_REFRESH_RATE = 10
    PROXY_ROTATION_RATE = 5
    DATA_DIR = Path("data")
    RESULTS_DIR = DATA_DIR / "results"
    INTERIM_DIR = DATA_DIR / "interim"
    
    @classmethod
    def initialize_directories(cls):
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.RESULTS_DIR.mkdir(exist_ok=True)
        cls.INTERIM_DIR.mkdir(exist_ok=True)

# Initialize directories
Config.initialize_directories()

# Data Management Classes
class DataManager:
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.interim_file = Config.INTERIM_DIR / f"interim_{self.session_id}.json"
        self.results = []
        self.processed_urls = set()
    
    def save_interim(self):
        """Save current results to interim storage"""
        try:
            with open(self.interim_file, 'w') as f:
                json.dump({
                    'results': self.results,
                    'processed_urls': list(self.processed_urls)
                }, f)
            logger.info(f"Saved interim results to {self.interim_file}")
        except Exception as e:
            logger.error(f"Failed to save interim results: {e}")
    
    def load_interim(self) -> bool:
        """Load results from interim storage if available"""
        try:
            if self.interim_file.exists():
                with open(self.interim_file, 'r') as f:
                    data = json.load(f)
                    self.results = data.get('results', [])
                    self.processed_urls = set(data.get('processed_urls', []))
                logger.info(f"Loaded interim results: {len(self.results)} entries")
                return True
        except Exception as e:
            logger.error(f"Failed to load interim results: {e}")
        return False
    
    def export_results(self, format: str = 'csv') -> str:
        """Export results in specified format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'csv':
            filename = Config.RESULTS_DIR / f"results_{timestamp}.csv"
            try:
                with open(filename, 'w', newline='') as f:
                    if self.results:
                        writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                        writer.writeheader()
                        writer.writerows(self.results)
                return str(filename)
            except Exception as e:
                logger.error(f"Failed to export CSV: {e}")
        
        elif format == 'json':
            filename = Config.RESULTS_DIR / f"results_{timestamp}.json"
            try:
                with open(filename, 'w') as f:
                    json.dump(self.results, f, indent=2)
                return str(filename)
            except Exception as e:
                logger.error(f"Failed to export JSON: {e}")
        
        return ""

class DataCleaner:
    @staticmethod
    def clean_text(text: str) -> str:
        """Basic text cleaning"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = text.strip()
        return text
    
    @staticmethod
    def normalize_phone(phone: str, country_code: str) -> str:
        """Normalize phone numbers to international format"""
        try:
            if not phone.startswith('+'):
                phone = f"+{phone}"
            parsed = phonenumbers.parse(phone, country_code)
            if phonenumbers.is_valid_number(parsed):
                return phonenumbers.format_number(
                    parsed,
                    phonenumbers.PhoneNumberFormat.INTERNATIONAL
                )
        except Exception as e:
            logger.debug(f"Phone normalization failed: {e}")
        return phone
    
    @staticmethod
    def normalize_email(email: str) -> str:
        """Normalize email addresses"""
        return email.lower().strip()
    
    @staticmethod
    def clean_company_name(name: str) -> str:
        """Clean and normalize company names"""
        if not name:
            return ""
        
        # Remove common suffixes
        suffixes = ['ltd', 'llc', 'inc', 'corp', 'corporation', 'co', 'company']
        name = name.lower()
        for suffix in suffixes:
            name = re.sub(f"\\b{suffix}\\b", "", name)
        
        # Clean up remaining text
        name = re.sub(r'[^\w\s-]', '', name)
        name = re.sub(r'\s+', ' ', name)
        return name.strip().title()
    
    @staticmethod
    def remove_duplicates(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entries based on key fields"""
        seen = set()
        cleaned = []
        
        for result in results:
            # Create unique key from cleaned essential fields
            key = (
                DataCleaner.clean_company_name(result.get('Name', '')),
                DataCleaner.normalize_email(result.get('Email', '')),
                DataCleaner.normalize_phone(result.get('Phone', ''), 'US')
            )
            
            if key not in seen and any(k for k in key):  # Ensure at least one field has value
                seen.add(key)
                cleaned.append(result)
        
        return cleaned

# Monitoring Classes
@dataclass
class ScrapingMetrics:
    start_time: datetime = None
    end_time: datetime = None
    total_urls: int = 0
    processed_urls: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    
    @property
    def duration(self) -> timedelta:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return timedelta(0)
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0
        return (self.successful_extractions / self.total_requests) * 100

class MonitoringSystem:
    def __init__(self):
        self.metrics = ScrapingMetrics()
        self.error_log = []
        self.lock = threading.Lock()
    
    def start_session(self):
        with self.lock:
            self.metrics = ScrapingMetrics(start_time=datetime.now())
    
    def end_session(self):
        with self.lock:
            self.metrics.end_time = datetime.now()
    
    def record_request(self, success: bool):
        with self.lock:
            self.metrics.total_requests += 1
            if not success:
                self.metrics.failed_requests += 1
    
    def record_extraction(self, success: bool):
        with self.lock:
            if success:
                self.metrics.successful_extractions += 1
            else:
                self.metrics.failed_extractions += 1
    
    def log_error(self, error: str, context: str = None):
        with self.lock:
            self.error_log.append({
                'timestamp': datetime.now().isoformat(),
                'error': str(error),
                'context': context
            })
    
    def get_metrics(self) -> Dict[str, Any]:
        with self.lock:
            return {
                'duration': str(self.metrics.duration),
                'total_requests': self.metrics.total_requests,
                'successful_extractions': self.metrics.successful_extractions,
                'failed_extractions': self.metrics.failed_extractions,
                'success_rate': f"{self.metrics.success_rate:.2f}%",
                'error_count': len(self.error_log)
            }

# Performance Optimization and Validation
class PerformanceOptimizer:
    def __init__(self):
        self.request_times = []
        self.extraction_times = []
        self.lock = threading.Lock()
        self.slow_threshold = 5.0  # seconds
    
    def record_request_time(self, duration: float):
        with self.lock:
            self.request_times.append(duration)
    
    def record_extraction_time(self, duration: float):
        with self.lock:
            self.extraction_times.append(duration)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        with self.lock:
            req_times = self.request_times[-100:] if self.request_times else []
            ext_times = self.extraction_times[-100:] if self.extraction_times else []
            
            return {
                'avg_request_time': statistics.mean(req_times) if req_times else 0,
                'avg_extraction_time': statistics.mean(ext_times) if ext_times else 0,
                'slow_requests': len([t for t in req_times if t > self.slow_threshold]),
                'total_processing_time': sum(req_times) + sum(ext_times)
            }

class ErrorTracker:
    def __init__(self):
        self.errors: Dict[str, List[Dict[str, Any]]] = {
            'network': [],
            'parsing': [],
            'validation': [],
            'timeout': [],
            'proxy': [],
            'other': []
        }
        self.lock = threading.Lock()
    
    def categorize_error(self, error: Exception) -> str:
        if isinstance(error, (requests.exceptions.ConnectionError, requests.exceptions.SSLError)):
            return 'network'
        elif isinstance(error, requests.exceptions.Timeout):
            return 'timeout'
        elif isinstance(error, ProxyError):
            return 'proxy'
        elif isinstance(error, (ValueError, AttributeError)):
            return 'parsing'
        elif isinstance(error, ValidationError):
            return 'validation'
        return 'other'
    
    def record_error(self, error: Exception, context: str = None):
        error_type = self.categorize_error(error)
        with self.lock:
            self.errors[error_type].append({
                'timestamp': datetime.now().isoformat(),
                'error': str(error),
                'context': context,
                'traceback': str(error.__traceback__)
            })
    
    def get_error_summary(self) -> Dict[str, Any]:
        with self.lock:
            return {
                category: len(errors)
                for category, errors in self.errors.items()
            }

class ValidationError(Exception):
    pass

class DataValidator:
    @staticmethod
    def validate_email(email: str) -> bool:
        if not email or email == 'N/A':
            return False
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_pattern, email))
    
    @staticmethod
    def validate_phone(phone: str, country_code: str) -> bool:
        try:
            if not phone or phone == 'N/A':
                return False
            parsed_number = phonenumbers.parse(phone, country_code)
            return phonenumbers.is_valid_number(parsed_number)
        except:
            return False
    
    @staticmethod
    def validate_url(url: str) -> bool:
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    @staticmethod
    def validate_company_name(name: str) -> bool:
        return bool(name and name != 'N/A' and len(name) >= 3)
    
    @staticmethod
    def calculate_confidence_score(result: Dict[str, Any]) -> float:
        score = 0
        if DataValidator.validate_email(result.get('Email')):
            score += 0.3
        if DataValidator.validate_phone(result.get('Phone'), 'US'):
            score += 0.3
        if DataValidator.validate_company_name(result.get('Name')):
            score += 0.2
        if DataValidator.validate_url(result.get('Website')):
            score += 0.2
        return score

class MetricsCollector:
    def __init__(self):
        self.metrics = {
            'requests': {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'timeouts': 0
            },
            'data': {
                'total_results': 0,
                'valid_emails': 0,
                'valid_phones': 0,
                'valid_companies': 0
            },
            'performance': {
                'total_time': 0,
                'avg_request_time': 0,
                'avg_processing_time': 0
            },
            'validation': {
                'high_confidence': 0,
                'medium_confidence': 0,
                'low_confidence': 0
            }
        }
        self.lock = threading.Lock()
    
    def update_request_metrics(self, success: bool, timeout: bool = False):
        with self.lock:
            self.metrics['requests']['total'] += 1
            if success:
                self.metrics['requests']['successful'] += 1
            else:
                self.metrics['requests']['failed'] += 1
            if timeout:
                self.metrics['requests']['timeouts'] += 1
    
    def update_data_metrics(self, result: Dict[str, Any]):
        with self.lock:
            self.metrics['data']['total_results'] += 1
            if DataValidator.validate_email(result.get('Email')):
                self.metrics['data']['valid_emails'] += 1
            if DataValidator.validate_phone(result.get('Phone'), 'US'):
                self.metrics['data']['valid_phones'] += 1
            if DataValidator.validate_company_name(result.get('Name')):
                self.metrics['data']['valid_companies'] += 1
    
    def update_validation_metrics(self, confidence_score: float):
        with self.lock:
            if confidence_score >= 0.8:
                self.metrics['validation']['high_confidence'] += 1
            elif confidence_score >= 0.5:
                self.metrics['validation']['medium_confidence'] += 1
            else:
                self.metrics['validation']['low_confidence'] += 1
    
    def update_performance_metrics(self, request_time: float, processing_time: float):
        with self.lock:
            self.metrics['performance']['total_time'] += request_time + processing_time
            self.metrics['performance']['avg_request_time'] = (
                self.metrics['performance']['avg_request_time'] * 
                (self.metrics['requests']['total'] - 1) +
                request_time
            ) / self.metrics['requests']['total']
            self.metrics['performance']['avg_processing_time'] = (
                self.metrics['performance']['avg_processing_time'] * 
                (self.metrics['requests']['total'] - 1) +
                processing_time
            ) / self.metrics['requests']['total']

# Initialize global instances
data_manager = DataManager()
monitoring_system = MonitoringSystem()
performance_optimizer = PerformanceOptimizer()
error_tracker = ErrorTracker()
metrics_collector = MetricsCollector()

# Enhanced scraping worker
def scraping_worker(country: str, keywords: List[str]):
    monitoring_system.start_session()
    start_time = time.time()
    
    try:
        # Load any existing interim results
        if data_manager.load_interim():
            logging.info(f"Resumed from {len(data_manager.results)} existing results")
        
        # Generate search queries
        search_queries = generate_search_queries(country, keywords)
        total_queries = len(search_queries)
        
        for query_index, query in enumerate(search_queries):
            if not scraping_status['is_running']:
                break
            
            query_start_time = time.time()
            logging.info(f"Processing query {query_index + 1}/{total_queries}: {query}")
            
            try:
                process_search_query(query, country)
            except Exception as e:
                error_tracker.record_error(e, f"Query: {query}")
                continue
            
            # Save interim results periodically
            if query_index % 5 == 0:
                data_manager.save_interim()
            
            query_duration = time.time() - query_start_time
            performance_optimizer.record_request_time(query_duration)
        
        # Final processing
        results = DataCleaner.remove_duplicates(data_manager.results)
        
        # Validate and score results
        validated_results = []
        for result in results:
            confidence_score = DataValidator.calculate_confidence_score(result)
            result['confidence_score'] = confidence_score
            metrics_collector.update_validation_metrics(confidence_score)
            if confidence_score >= 0.5:  # Only keep medium and high confidence results
                validated_results.append(result)
        
        # Export results
        data_manager.results = validated_results
        data_manager.export_results('csv')
        data_manager.export_results('json')
        
    except Exception as e:
        error_tracker.record_error(e, "Main scraping process")
        logging.error(f"Scraping error: {str(e)}")
    finally:
        monitoring_system.end_session()
        total_duration = time.time() - start_time
        performance_optimizer.record_extraction_time(total_duration)

# Request handling and data extraction functions
class RequestHandler:
    def __init__(self, country_code: str):
        self.country_code = country_code
        self.request_manager = RequestManager()
        self.total_requests = 0
        self.successful_requests = 0
    
    @retry(
        stop=stop_after_attempt(Config.RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def make_request(self, url: str) -> Optional[BeautifulSoup]:
        try:
            start_time = time.time()
            self.total_requests += 1
            
            response = self.request_manager.make_request(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            duration = time.time() - start_time
            performance_optimizer.record_request_time(duration)
            metrics_collector.update_request_metrics(True)
            self.successful_requests += 1
            
            return soup
        
        except Exception as e:
            error_tracker.record_error(e, f"URL: {url}")
            metrics_collector.update_request_metrics(False, isinstance(e, requests.Timeout))
            raise

def process_search_query(query: str, country: str):
    request_handler = RequestHandler(country)
    
    for page in range(1, 4):  # Search first 3 pages
        if not scraping_status['is_running']:
            break
        
        start_time = time.time()
        try:
            urls = get_google_search_results(query, page)
            for url in urls:
                if url in data_manager.processed_urls:
                    continue
                
                result = process_single_url(url, country, request_handler)
                if result:
                    data_manager.results.append(result)
                    data_manager.processed_urls.add(url)
                    metrics_collector.update_data_metrics(result)
        
        except Exception as e:
            error_tracker.record_error(e, f"Query: {query}, Page: {page}")
        
        duration = time.time() - start_time
        performance_optimizer.record_extraction_time(duration)
        time.sleep(Config.RATE_LIMIT_DELAY)

def process_single_url(url: str, country: str, request_handler: RequestHandler) -> Optional[Dict[str, Any]]:
    try:
        soup = request_handler.make_request(url)
        if not soup:
            return None
        
        result = extract_contact_details(soup, url, country)
        if result:
            confidence_score = DataValidator.calculate_confidence_score(result)
            result['confidence_score'] = confidence_score
            metrics_collector.update_validation_metrics(confidence_score)
            
            if confidence_score >= 0.5:
                return result
        
        return None
    
    except Exception as e:
        error_tracker.record_error(e, f"URL Processing: {url}")
        return None

# Flask routes and API endpoints
@app.route('/')
def home():
    return render_template('index.html',
                         industry_keywords=INDUSTRY_KEYWORDS,
                         business_keywords=BUSINESS_KEYWORDS)

@app.route('/start_scraping', methods=['POST'])
def start_scraping():
    try:
        data = request.json
        country = data.get('country')
        keywords = data.get('keywords', '').split('\n')
        
        if not country:
            return jsonify({'error': 'Missing country'}), 400
        
        # Validate country code
        country = country.upper()
        if not re.match(r'^[A-Z]{2}$', country):
            return jsonify({'error': 'Invalid country code format'}), 400
        
        # Reset status and start new scraping session
        global scraping_status
        scraping_status = {
            'is_running': True,
            'progress': 0,
            'total': 0,
            'current_status': 'Starting...',
            'results': [],
            'errors': []
        }
        
        # Start scraping in background thread
        thread = threading.Thread(
            target=scraping_worker,
            args=(country, keywords)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({'message': 'Scraping started successfully'})
    
    except Exception as e:
        error_tracker.record_error(e, "Start scraping endpoint")
        return jsonify({'error': str(e)}), 500

@app.route('/stop_scraping', methods=['POST'])
def stop_scraping():
    global scraping_status
    scraping_status['is_running'] = False
    return jsonify({'message': 'Scraping stopped'})

@app.route('/status')
def get_status():
    return jsonify({
        **scraping_status,
        'metrics': metrics_collector.metrics,
        'performance': performance_optimizer.get_performance_metrics(),
        'errors': error_tracker.get_error_summary()
    })

@app.route('/download/<format>')
def download_results(format):
    if format not in ['csv', 'json']:
        return jsonify({'error': 'Invalid format'}), 400
    
    if not data_manager.results:
        return jsonify({'error': 'No results available'}), 404
    
    try:
        filename = data_manager.export_results(format)
        if not filename:
            return jsonify({'error': 'Export failed'}), 500
        
        return send_file(
            filename,
            as_attachment=True,
            download_name=f'scraping_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{format}'
        )
    
    except Exception as e:
        error_tracker.record_error(e, f"Download {format} endpoint")
        return jsonify({'error': str(e)}), 500

@app.route('/metrics')
def get_metrics():
    return jsonify({
        'monitoring': monitoring_system.get_metrics(),
        'performance': performance_optimizer.get_performance_metrics(),
        'errors': error_tracker.get_error_summary(),
        'data': metrics_collector.metrics
    })

@app.route('/errors')
def get_errors():
    category = request.args.get('category', 'all')
    if category == 'all':
        return jsonify(error_tracker.errors)
    return jsonify(error_tracker.errors.get(category, []))

# Main entry point
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
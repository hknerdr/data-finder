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
import json
import io
import csv
import os
from fake_useragent import UserAgent
import itertools
from tenacity import retry, stop_after_attempt, wait_exponential
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import statistics
import traceback
from werkzeug.exceptions import HTTPException
import sys
import pycountry
import nltk  # Added import

# Initialize NLTK (ensure it's imported)
try:
    nltk.data.path.append('/opt/render/nltk_data')
    nltk.download('punkt', quiet=True, download_dir='/opt/render/nltk_data')
except Exception as e:
    print(f"NLTK initialization warning (non-critical): {e}")

# Configure logging to work with Render.com
BASE_DIR = Path(os.getenv('RENDER_APP_DIR', os.getcwd()))
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Create necessary directories with proper permissions
for directory in [DATA_DIR, LOGS_DIR]:
    directory.mkdir(mode=0o755, parents=True, exist_ok=True)

# Configure logging
log_file = os.path.join(os.getenv('RENDER_APP_DIR', os.getcwd()), 'scraper.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)  # Log to stdout for Render.com to capture
    ]
)

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialization Middleware
class InitializationMiddleware:
    def __init__(self, app):
        self.app = app
        self._initialized = False
        self._init_lock = threading.Lock()

    def __call__(self, environ, start_response):
        if not self._initialized:
            with self._init_lock:
                if not self._initialized:
                    try:
                        Config.initialize_directories()
                        global_state.initialize()
                        logger.info("Application initialized successfully")
                        self._initialized = True
                    except Exception as e:
                        logger.error(f"Failed to initialize application: {e}\n{traceback.format_exc()}")
                        raise
        return self.app(environ, start_response)

# Apply middleware
app.wsgi_app = InitializationMiddleware(app.wsgi_app)

# Configuration and Keywords
class Keywords:
    INDUSTRY_KEYWORDS = [
        'Non-Alcoholic Beverages', 'Soft Drinks', 'Beverages', 'Juices', 'Mineral Water',
        'Bottled Water', 'Energy Drinks', 'Sports Drinks', 'Carbonated Drinks',
        'Flavored Water', 'Herbal Drinks', 'Functional Beverages', 'Dairy Beverages',
        'Plant-Based Beverages', 'Smoothies', 'Iced Tea', 'Ready-to-Drink Coffee',
        'Mocktails', 'Kombucha', 'Vitamin Water'
    ]

    BUSINESS_KEYWORDS = [
        'Distributor', 'Wholesaler', 'Supplier', 'Trader', 'Dealer', 'Reseller',
        'Stockist', 'Merchant', 'Importer', 'Exporter', 'Agency', 'Broker',
        'Trading Company', 'Bulk Supplier', 'B2B Supplier'
    ]

class Config:
    RETRY_ATTEMPTS = 3
    REQUEST_TIMEOUT = 15  # Reduced timeout for Render.com
    RATE_LIMIT_DELAY = 2
    MAX_RESULTS_PER_QUERY = 10  # Reduced for stability
    USER_AGENT_REFRESH_RATE = 5
    
    # Use environment variables
    BASE_DIR = Path(os.getenv('RENDER_APP_DIR', os.getcwd()))
    DATA_DIR = BASE_DIR / "data"
    RESULTS_DIR = DATA_DIR / "results"
    INTERIM_DIR = DATA_DIR / "interim"
    
    @classmethod
    def initialize_directories(cls):
        try:
            for directory in [cls.DATA_DIR, cls.RESULTS_DIR, cls.INTERIM_DIR]:
                directory.mkdir(mode=0o755, parents=True, exist_ok=True)
            logger.info("Directories initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize directories: {e}")
            # Continue even if directory creation fails
            pass

# Define SearchEngine once before GlobalState
class SearchEngine:
    def __init__(self):
        self.session = requests.Session()
        self.user_agent = UserAgent()
    
    def search(self, query: str, country_code: str) -> List[str]:
        try:
            headers = {
                'User-Agent': self.user_agent.random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            # Example using DuckDuckGo (you can modify for other search engines)
            params = {
                'q': query,
                'kl': f'region:{country_code}',
                'format': 'json'
            }
            
            response = self.session.get(
                'https://api.duckduckgo.com/',
                params=params,
                headers=headers,
                timeout=10
            )
            
            # Process results (modify according to the search engine's response format)
            if response.status_code == 200:
                results = response.json()
                return [result['url'] for result in results.get('Results', [])]
            
            return []
                
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

# Global state
class GlobalState:
    def __init__(self):
        self.scraping_status = {
            'is_running': False,
            'progress': 0,
            'total': 0,
            'current_status': 'Ready',
            'results': [],
            'errors': []
        }
        self.data_manager = None
        self.monitoring_system = None
        self.performance_optimizer = None
        self.error_tracker = None
        self.metrics_collector = None
        self.search_engine = SearchEngine()  # Now correctly defined
    
    def initialize(self):
        try:
            self.data_manager = DataManager()
            self.monitoring_system = MonitoringSystem()
            self.performance_optimizer = PerformanceOptimizer()
            self.error_tracker = ErrorTracker()
            self.metrics_collector = MetricsCollector()
            logger.info("Global state initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize global state: {e}")
            raise

global_state = GlobalState()

# Data Management Classes
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

class DataManager:
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.interim_file = Config.INTERIM_DIR / f"interim_{self.session_id}.json"
        self.results = []
        self.processed_urls = set()
    
    def save_interim(self):
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            if format == 'csv':
                filename = Config.RESULTS_DIR / f"results_{timestamp}.csv"
                with open(filename, 'w', newline='') as f:
                    if self.results:
                        writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                        writer.writeheader()
                        writer.writerows(self.results)
                return str(filename)
            elif format == 'json':
                filename = Config.RESULTS_DIR / f"results_{timestamp}.json"
                with open(filename, 'w') as f:
                    json.dump(self.results, f, indent=2)
                return str(filename)
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            return ""

class DataCleaner:
    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        # Simple text cleaning without NLTK
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def normalize_phone(phone: str, country_code: str) -> str:
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
        return email.lower().strip()
    
    @staticmethod
    def clean_company_name(name: str) -> str:
        if not name:
            return ""
        suffixes = ['ltd', 'llc', 'inc', 'corp', 'corporation', 'co', 'company']
        name = name.lower()
        for suffix in suffixes:
            name = re.sub(f"\\b{suffix}\\b", "", name)
        name = re.sub(r'[^\w\s-]', '', name)
        name = re.sub(r'\s+', ' ', name)
        return name.strip().title()
    
    @staticmethod
    def remove_duplicates(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        cleaned = []
        
        for result in results:
            key = (
                DataCleaner.clean_company_name(result.get('Name', '')),
                DataCleaner.normalize_email(result.get('Email', '')),
                DataCleaner.normalize_phone(result.get('Phone', ''), 'US')
            )
            if key not in seen and any(k for k in key):
                seen.add(key)
                cleaned.append(result)
        return cleaned

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
            'other': []
        }
        self.lock = threading.Lock()
    
    def categorize_error(self, error: Exception) -> str:
        if isinstance(error, requests.exceptions.ConnectionError):
            return 'network'
        elif isinstance(error, requests.exceptions.Timeout):
            return 'timeout'
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
                'traceback': ''.join(traceback.format_exception(None, error, error.__traceback__))
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
        self.metrics = {}
        self.lock = threading.Lock()

    def update_metrics(self, new_metrics: Dict[str, Any]):
        with self.lock:
            self.metrics.update(new_metrics)

# The SearchEngine class is already defined above

# Request Manager Class (assuming it's defined elsewhere or needs to be added)
class RequestManager:
    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()
        self.request_count = 0
    
    def get_headers(self) -> Dict[str, str]:
        self.request_count += 1
        if self.request_count % Config.USER_AGENT_REFRESH_RATE == 0:
            self.ua = UserAgent()
        
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    @retry(
        stop=stop_after_attempt(Config.RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def make_request(self, url: str, method: str = 'get', **kwargs) -> Optional[requests.Response]:
        try:
            kwargs.update({
                'headers': self.get_headers(),
                'timeout': Config.REQUEST_TIMEOUT,
                'verify': True,
                'allow_redirects': True
            })
            
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            
            time.sleep(random.uniform(1.0, 2.0))
            
            return response
            
        except Exception as e:
            logger.error(f"Request failed for {url}: {str(e)}")
            raise  # To trigger retry

# Error Handler
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}\n{traceback.format_exc()}")
    if isinstance(e, HTTPException):
        return e
    
    error_details = {
        'error': str(e),
        'type': e.__class__.__name__,
        'trace_id': datetime.now().strftime('%Y%m%d%H%M%S')
    }
    
    if app.debug:
        error_details['traceback'] = traceback.format_exc()
    
    return jsonify(error_details), 500

# Home Route
@app.route('/')
def home():
    try:
        # Get list of countries
        countries = [(country.alpha_2, country.name) 
                    for country in pycountry.countries]
        countries.sort(key=lambda x: x[1])  # Sort by country name
        
        return render_template('index.html',
                             countries=countries,
                             industry_keywords=Keywords.INDUSTRY_KEYWORDS,
                             business_keywords=Keywords.BUSINESS_KEYWORDS)
    except Exception as e:
        logger.error(f"Error in home route: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

# Define the generate_search_queries function
def generate_search_queries(country: str, keywords: List[str] = None) -> List[str]:
    """
    Generate a list of search queries based on the selected country and keywords.
    If no keywords are provided, use default industry and business keyword combinations.
    """
    if not keywords:
        # Generate combinations of industry and business keywords
        keywords = [f"{industry} {business}" for industry in Keywords.INDUSTRY_KEYWORDS for business in Keywords.BUSINESS_KEYWORDS]
    
    # Optionally, incorporate the country into the search queries
    queries = [f"{keyword} {country}" for keyword in keywords]
    return queries

# Scraping Worker Function
def scraping_worker(country: str, keywords: List[str] = None):
    """Worker function for scraping process"""
    try:
        queries = generate_search_queries(country, keywords)
        global_state.scraping_status['total'] = len(queries)
        
        global_state.monitoring_system.start_session()
        logger.info(f"Starting scraping for country: {country} with {len(queries)} queries")
        
        request_manager = RequestManager()
        
        for idx, query in enumerate(queries, 1):
            if not global_state.scraping_status['is_running']:
                logger.info("Scraping stopped by user")
                break
            
            try:
                status_message = f"Processing query {idx}/{len(queries)}: {query}"
                logger.info(status_message)
                global_state.scraping_status.update({
                    'current_status': status_message,
                    'progress': idx
                })
                
                # Use multiple search domains to avoid blocking
                search_domains = [
                    "www.bing.com/search",
                    "search.yahoo.com/search",
                    "www.ecosia.org/search"
                ]
                
                for search_domain in search_domains:
                    try:
                        search_url = f"https://{search_domain}?q={quote(query)}"
                        response = request_manager.make_request(search_url)
                        
                        if response and response.status_code == 200:
                            # Process results...
                            # Placeholder for actual scraping logic
                            logger.debug(f"Successfully fetched data from {search_domain}")
                            time.sleep(random.uniform(2.0, 4.0))
                            break  # Successfully got results, move to next query
                        
                    except Exception as e:
                        logger.error(f"Search failed for {search_domain}: {str(e)}")
                        continue
                
                time.sleep(random.uniform(3.0, 5.0))  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                time.sleep(random.uniform(5.0, 10.0))  # Longer delay on errors
                continue
            
        # Cleanup
        global_state.scraping_status.update({
            'is_running': False,
            'current_status': "Completed",
            'progress': len(queries)
        })
        logger.info("Scraping process completed")
        
    except Exception as e:
        logger.error(f"Scraping worker failed: {str(e)}")
        global_state.scraping_status.update({
            'current_status': f"Failed: {str(e)}",
            'is_running': False
        })

# Health Check Route
@app.route('/health')
def health_check():
    try:
        # Basic health checks
        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'directory_status': {},
            'memory_usage': os.getenv('RENDER_MEMORY_MB', 'unknown'),
            'scraping_status': global_state.scraping_status['current_status']
        }
        
        # Check directories
        for dir_name, dir_path in {
            'data': Config.DATA_DIR,
            'results': Config.RESULTS_DIR,
            'interim': Config.INTERIM_DIR
        }.items():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                test_file = dir_path / '.test'
                test_file.touch()
                test_file.unlink()
                status['directory_status'][dir_name] = 'writable'
            except Exception as e:
                status['directory_status'][dir_name] = f'error: {str(e)}'
        
        return jsonify(status)
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

# Start Scraping Route
@app.route('/start_scraping', methods=['POST'])
def start_scraping():
    try:
        data = request.json
        if not data:
            logger.error("No JSON data received")
            return jsonify({'error': 'No data received'}), 400
        
        logger.debug(f"Received data: {data}")
        
        country = data.get('country')
        if not country:
            return jsonify({'error': 'Missing country'}), 400
        
        keywords = data.get('keywords', '')
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split('\n') if k.strip()]
        elif not isinstance(keywords, list):
            keywords = []
        
        # Reset status and start new scraping session
        global_state.scraping_status.update({
            'is_running': True,
            'progress': 0,
            'total': 0,
            'current_status': 'Starting...',
            'results': [],
            'errors': []
        })
        
        # Start scraping thread
        thread = threading.Thread(
            target=scraping_worker,
            args=(country, keywords)
        )
        thread.daemon = True
        thread.start()
        
        logger.info(f"Scraping started for country: {country} with {len(keywords)} keywords")
        return jsonify({'message': 'Scraping started successfully'})
        
    except Exception as e:
        logger.error(f"Error in start_scraping: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'error': str(e),
            'type': e.__class__.__name__
        }), 500

# Stop Scraping Route
@app.route('/stop_scraping', methods=['POST'])
def stop_scraping():
    try:
        global_state.scraping_status['is_running'] = False
        return jsonify({'message': 'Scraping stopped'})
    except Exception as e:
        logger.error(f"Error stopping scraping: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Status Route
@app.route('/status')
def get_status():
    try:
        return jsonify({
            **global_state.scraping_status,
            'metrics': global_state.metrics_collector.metrics if global_state.metrics_collector else {},
            'performance': global_state.performance_optimizer.get_performance_metrics() if global_state.performance_optimizer else {},
            'errors': global_state.error_tracker.get_error_summary() if global_state.error_tracker else {}
        })
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Download Results Route
@app.route('/download/<format>')
def download_results(format):
    try:
        if format not in ['csv', 'json']:
            return jsonify({'error': 'Invalid format'}), 400
        
        if not global_state.data_manager or not global_state.data_manager.results:
            return jsonify({'error': 'No results available'}), 404
        
        filename = global_state.data_manager.export_results(format)
        if not filename:
            return jsonify({'error': 'Export failed'}), 500
        
        try:
            return send_file(
                filename,
                as_attachment=True,
                download_name=f'scraping_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{format}'
            )
        except Exception as e:
            logger.error(f"Error sending file: {str(e)}")
            return jsonify({'error': 'File delivery failed'}), 500
            
    except Exception as e:
        logger.error(f"Error downloading results: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Metrics Route
@app.route('/metrics')
def get_metrics():
    try:
        return jsonify({
            'monitoring': global_state.monitoring_system.get_metrics() if global_state.monitoring_system else {},
            'performance': global_state.performance_optimizer.get_performance_metrics() if global_state.performance_optimizer else {},
            'errors': global_state.error_tracker.get_error_summary() if global_state.error_tracker else {},
            'data': global_state.metrics_collector.metrics if global_state.metrics_collector else {}
        })
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Errors Route
@app.route('/errors')
def get_errors():
    try:
        category = request.args.get('category', 'all')
        if not global_state.error_tracker:
            return jsonify({'error': 'Error tracker not initialized'}), 500
        
        if category == 'all':
            return jsonify(global_state.error_tracker.errors)
        return jsonify(global_state.error_tracker.errors.get(category, []))
    except Exception as e:
        logger.error(f"Error getting errors: {str(e)}")
        return jsonify({'error': str(e)}), 500

def create_app():
    return app

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)

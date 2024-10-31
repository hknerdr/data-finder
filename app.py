# app_part1.py

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
import traceback
from werkzeug.exceptions import HTTPException
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler(sys.stdout)
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
    REQUEST_TIMEOUT = 30
    RATE_LIMIT_DELAY = 2
    MAX_RESULTS_PER_QUERY = 20
    USER_AGENT_REFRESH_RATE = 10
    PROXY_ROTATION_RATE = 5
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = BASE_DIR / "data"
    RESULTS_DIR = DATA_DIR / "results"
    INTERIM_DIR = DATA_DIR / "interim"
    
    @classmethod
    def initialize_directories(cls):
        try:
            cls.DATA_DIR.mkdir(exist_ok=True)
            cls.RESULTS_DIR.mkdir(exist_ok=True)
            cls.INTERIM_DIR.mkdir(exist_ok=True)
            logger.info("Directories initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize directories: {e}")
            raise

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

# Initialize global state
global_state = GlobalState()

# Initialize directories
Config.initialize_directories()

# app_part2.py

from datetime import datetime, timedelta
import json
import csv
from typing import Dict, List, Any
from dataclasses import dataclass
import re
import phonenumbers
import logging
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

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
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
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

# app_part3.py

import threading
import statistics
from datetime import datetime
import requests
from requests.exceptions import RequestException, ProxyError
from typing import Dict, List, Any
import logging
from urllib.parse import urlparse
import phonenumbers
import re

logger = logging.getLogger(__name__)

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
        self.metrics = {}
        self.lock = threading.Lock()

    def update_metrics(self, new_metrics: Dict[str, Any]):
        with self.lock:
            self.metrics.update(new_metrics)

# app_part4.py

from flask import Flask, render_template, request, jsonify, send_file
import logging
from datetime import datetime
import threading
import traceback
from werkzeug.exceptions import HTTPException
import os
import sys

logger = logging.getLogger(__name__)

# Import the app instance from part 1
from app_part1 import app, global_state, Config

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

@app.route('/health')
def health_check():
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'scraping_status': global_state.scraping_status['current_status'],
            'data_dir_exists': Config.DATA_DIR.exists(),
            'results_dir_exists': Config.RESULTS_DIR.exists(),
            'interim_dir_exists': Config.INTERIM_DIR.exists()
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/')
def home():
    try:
        return render_template('index.html',
                             industry_keywords=Keywords.INDUSTRY_KEYWORDS,
                             business_keywords=Keywords.BUSINESS_KEYWORDS)
    except Exception as e:
        logger.error(f"Error in home route: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

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

@app.route('/stop_scraping', methods=['POST'])
def stop_scraping():
    try:
        global_state.scraping_status['is_running'] = False
        return jsonify({'message': 'Scraping stopped'})
    except Exception as e:
        logger.error(f"Error stopping scraping: {str(e)}")
        return jsonify({'error': str(e)}), 500

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
        
        return send_file(
            filename,
            as_attachment=True,
            download_name=f'scraping_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{format}'
        )
    except Exception as e:
        logger.error(f"Error downloading results: {str(e)}")
        return jsonify({'error': str(e)}), 500

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

# Main entry point
if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)
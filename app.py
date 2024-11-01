# File: app.py

from flask import Flask, render_template, request, jsonify, send_file
import requests
from bs4 import BeautifulSoup
import threading
import time
import re
import phonenumbers
import logging
from urllib.parse import urlparse, quote_plus, parse_qs
from datetime import datetime, timedelta
import json
import csv
import os
from fake_useragent import UserAgent
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

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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

@dataclass
class Proxy:
    protocol: str
    host: str
    port: int
    username: str
    password: str

    def get_url(self) -> str:
        if self.username and self.password:
            return f"{self.protocol}://{self.username}:{self.password}@{self.host}:{self.port}"
        return f"{self.protocol}://{self.host}:{self.port}"

    def to_dict(self) -> Dict[str, str]:
        proxy_url = self.get_url()
        return {
            'http': proxy_url,
            'https': proxy_url
        }

class DataManager:
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.interim_file = Path("data/interim") / f"interim_{self.session_id}.json"
        self.results = []
        self.processed_urls = set()

    def save_interim(self):
        try:
            self.interim_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.interim_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'results': self.results,
                    'processed_urls': list(self.processed_urls)
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved interim results to {self.interim_file}")
        except Exception as e:
            logger.error(f"Failed to save interim results: {e}")

    def export_results(self, format: str = 'csv') -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            if format == 'csv':
                filename = Path("data/results") / f"results_{timestamp}.csv"
                filename.parent.mkdir(parents=True, exist_ok=True)
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    if self.results:
                        fieldnames = ['URL', 'Emails', 'Phones']
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for result in self.results:
                            writer.writerow({
                                'URL': result.get('URL', ''),
                                'Emails': ', '.join(result.get('Emails', [])),
                                'Phones': ', '.join(result.get('Phones', []))
                            })
                return str(filename)
            elif format == 'json':
                filename = Path("data/results") / f"results_{timestamp}.json"
                filename.parent.mkdir(parents=True, exist_ok=True)
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, indent=2, ensure_ascii=False)
                return str(filename)
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            return ""

class ScrapingMetrics:
    def __init__(self, start_time: Optional[datetime] = None):
        self.start_time = start_time
        self.end_time: Optional[datetime] = None
        self.total_requests: int = 0
        self.failed_requests: int = 0
        self.successful_extractions: int = 0
        self.failed_extractions: int = 0

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
        email_pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
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
        if result.get('Emails') and DataValidator.validate_email(result.get('Emails')[0]):
            score += 0.3
        if result.get('Phones') and DataValidator.validate_phone(result.get('Phones')[0], 'US'):
            score += 0.3
        if DataValidator.validate_company_name(result.get('Name', '')):
            score += 0.2
        if DataValidator.validate_url(result.get('URL', '')):
            score += 0.2
        return score

class MetricsCollector:
    def __init__(self):
        self.metrics = {}
        self.lock = threading.Lock()

    def update_metrics(self, new_metrics: Dict[str, Any]):
        with self.lock:
            self.metrics.update(new_metrics)

class SearchEngine:
    def __init__(self, state: 'GlobalState'):
        self.state = state
        self.session = requests.Session()
        self.user_agent = UserAgent()

    def search(self, query: str) -> List[str]:
        try:
            headers = {
                'User-Agent': self.user_agent.random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            params = {'q': query, 'hl': 'en', 'num': 10}
            proxies = self.get_proxies()

            response = self.session.get(
                'https://www.google.com/search',
                params=params,
                headers=headers,
                proxies=proxies,
                timeout=15
            )

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                results = []
                for a_tag in soup.find_all('a'):
                    href = a_tag.get('href')
                    if href and href.startswith('/url?q='):
                        url = parse_qs(urlparse(href).query).get('q')
                        if url:
                            clean_url = url[0].split('&')[0]
                            if clean_url.startswith('http'):
                                results.append(clean_url)
                return results
            return []
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_proxies(self) -> Optional[Dict[str, str]]:
        if not self.state.proxies:
            return None
        proxy = random.choice(self.state.proxies)
        return proxy.to_dict()

# Proxy Validation Function
def validate_proxy(proxy: Proxy) -> bool:
    try:
        proxies = proxy.to_dict()
        response = requests.get("https://httpbin.org/ip", proxies=proxies, timeout=10)
        return response.status_code == 200
    except:
        return False

# Scraping Worker Function
def scraping_worker(country: str, keywords: List[str] = None):
    try:
        queries = generate_search_queries(country, keywords)
        with global_state.lock:
            global_state.scraping_status['total'] = len(queries)

        global_state.monitoring_system.start_session()
        logger.info(f"Starting scraping for country: {country} with {len(queries)} queries")

        request_manager = RequestManager()

        for idx, query in enumerate(queries, 1):
            with global_state.lock:
                if not global_state.scraping_status['is_running']:
                    logger.info("Scraping stopped by user")
                    break

            try:
                status_message = f"Processing query {idx}/{len(queries)}: {query}"
                logger.info(status_message)
                with global_state.lock:
                    global_state.scraping_status['current_status'] = status_message
                    global_state.scraping_status['progress'] = idx

                results = global_state.search_engine.search(query)

                if not results:
                    logger.warning(f"No results found for query: {query}")

                for url in results:
                    with global_state.lock:
                        if url in global_state.data_manager.processed_urls:
                            continue
                        global_state.data_manager.processed_urls.add(url)

                    contacts = extract_contacts_from_url(url, request_manager, country_code=country)
                    if contacts['Emails'] or contacts['Phones']:
                        with global_state.lock:
                            global_state.scraping_status['results'].append({
                                'URL': url,
                                'Emails': contacts['Emails'],
                                'Phones': contacts['Phones']
                            })
                        logger.debug(f"Extracted contacts from {url}: {contacts}")
                    else:
                        with global_state.lock:
                            global_state.scraping_status['results'].append({
                                'URL': url,
                                'Emails': [],
                                'Phones': []
                            })

                time.sleep(random.uniform(2.0, 4.0))

            except Exception as e:
                logger.error(f"Error processing query: {e}")
                with global_state.lock:
                    global_state.scraping_status['errors'].append(str(e))
                time.sleep(random.uniform(5.0, 10.0))
                continue

            if idx % 50 == 0:
                with global_state.lock:
                    global_state.data_manager.results = global_state.scraping_status['results']
                global_state.data_manager.save_interim()

        with global_state.lock:
            cleaned_results = DataCleaner.remove_duplicates(global_state.scraping_status['results'])
            global_state.scraping_status['results'] = cleaned_results

        global_state.data_manager.results = cleaned_results
        global_state.data_manager.save_interim()

        with global_state.lock:
            global_state.scraping_status.update({
                'is_running': False,
                'current_status': "Completed",
                'progress': len(queries)
            })
        logger.info("Scraping process completed")

    except Exception as e:
        logger.error(f"Scraping worker failed: {e}")
        with global_state.lock:
            global_state.scraping_status.update({
                'current_status': f"Failed: {e}",
                'is_running': False
            })

# Utility Functions
def generate_search_queries(country: str, keywords: List[str] = None) -> List[str]:
    if not keywords:
        keywords = [f"{industry} {business}" for industry in Keywords.INDUSTRY_KEYWORDS for business in Keywords.BUSINESS_KEYWORDS]
    queries = [f"{keyword} {country}" for keyword in keywords]
    return queries

EMAIL_REGEX = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
PHONE_REGEX = r'\+?\d[\d -]{8,}\d'

def extract_emails(text):
    return re.findall(EMAIL_REGEX, text)

def extract_phone_numbers(text, country_code='US'):
    phones = []
    for match in re.findall(PHONE_REGEX, text):
        try:
            parsed = phonenumbers.parse(match, country_code)
            if phonenumbers.is_valid_number(parsed):
                formatted = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
                phones.append(formatted)
        except:
            continue
    return phones

def extract_contacts_from_url(url: str, request_manager: 'RequestManager', country_code: str = 'US') -> Dict[str, List[str]]:
    emails = []
    phones = []
    try:
        response = request_manager.make_request(url)
        if response and response.status_code == 200:
            emails = extract_emails(response.text)
            phones = extract_phone_numbers(response.text, country_code)
            emails = [email.lower().strip() for email in emails if DataValidator.validate_email(email)]
            phones = [phone.strip() for phone in phones if DataValidator.validate_phone(phone, country_code)]
            emails = list(set(emails))
            phones = list(set(phones))
    except Exception as e:
        logger.error(f"Failed to extract contacts from {url}: {e}")
    return {'Emails': emails, 'Phones': phones}

class RequestManager:
    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()
        self.request_count = 0

    def get_headers(self) -> Dict[str, str]:
        self.request_count += 1
        if self.request_count % 5 == 0:
            self.ua = UserAgent()
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def make_request(self, url: str, method: str = 'get', **kwargs) -> Optional[requests.Response]:
        try:
            headers = self.get_headers()
            proxies = global_state.search_engine.get_proxies()
            kwargs.update({
                'headers': headers,
                'timeout': 15,
                'proxies': proxies,
                'allow_redirects': True
            })
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except Exception as e:
            logger.error(f"Request failed for {url}: {e}")
            raise

class DataCleaner:
    @staticmethod
    def remove_duplicates(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_urls = set()
        cleaned = []
        for result in results:
            url = result.get('URL')
            if url and url not in seen_urls:
                seen_urls.add(url)
                cleaned.append(result)
        return cleaned

# Global State Class
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
        self.proxies: List[Proxy] = []
        self.data_manager = DataManager()
        self.lock = threading.Lock()
        self.search_engine = SearchEngine(self)
        self.monitoring_system = MonitoringSystem()
        self.performance_optimizer = PerformanceOptimizer()
        self.error_tracker = ErrorTracker()
        self.metrics_collector = MetricsCollector()

global_state = GlobalState()

# Flask Routes
@app.route('/')
def home():
    try:
        countries = [(country.alpha_2, country.name) for country in pycountry.countries]
        countries.sort(key=lambda x: x[1])
        return render_template('index.html', countries=countries,
                               industry_keywords=Keywords.INDUSTRY_KEYWORDS,
                               business_keywords=Keywords.BUSINESS_KEYWORDS)
    except Exception as e:
        logger.error(f"Error in home route: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/add_proxies', methods=['POST'])
def add_proxies():
    try:
        data = request.json
        proxies = data.get('proxies', [])
        new_proxies = []
        for p in proxies:
            proxy = Proxy(protocol=p['protocol'],
                          host=p['host'],
                          port=int(p['port']),
                          username=p.get('username', ''),
                          password=p.get('password', ''))
            new_proxies.append(proxy)
        with global_state.lock:
            global_state.proxies.extend(new_proxies)
        return jsonify({'message': 'Proxies added successfully'}), 200
    except Exception as e:
        logger.error(f"Error adding proxies: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/validate_proxies', methods=['POST'])
def validate_proxies():
    try:
        with global_state.lock:
            proxies = global_state.proxies.copy()
        results = []
        for proxy in proxies:
            is_valid = validate_proxy(proxy)
            results.append({
                'proxy': proxy.get_url(),
                'valid': is_valid
            })
        return jsonify({'validation_results': results}), 200
    except Exception as e:
        logger.error(f"Error validating proxies: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/start_scraping', methods=['POST'])
def start_scraping():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data received'}), 400
        country = data.get('country')
        if not country:
            return jsonify({'error': 'Missing country'}), 400
        keywords = data.get('keywords', '')
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split('\n') if k.strip()]
        elif not isinstance(keywords, list):
            keywords = []

        with global_state.lock:
            global_state.scraping_status.update({
                'is_running': True,
                'progress': 0,
                'total': 0,
                'current_status': 'Starting...',
                'results': [],
                'errors': []
            })
            global_state.data_manager.results = []
            global_state.data_manager.processed_urls = set()

        thread = threading.Thread(target=scraping_worker, args=(country, keywords))
        thread.daemon = True
        thread.start()

        return jsonify({'message': 'Scraping started successfully'}), 200
    except Exception as e:
        logger.error(f"Error in start_scraping: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stop_scraping', methods=['POST'])
def stop_scraping():
    try:
        with global_state.lock:
            global_state.scraping_status['is_running'] = False
        return jsonify({'message': 'Scraping stopped'}), 200
    except Exception as e:
        logger.error(f"Error stopping scraping: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def get_status():
    try:
        with global_state.lock:
            return jsonify({
                **global_state.scraping_status,
                'metrics': global_state.monitoring_system.get_metrics(),
                'performance': global_state.performance_optimizer.get_performance_metrics(),
                'errors': global_state.error_tracker.get_error_summary()
            })
    except Exception as e:
        logger.error(f"Error getting status: {e}")
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
        return send_file(filename, as_attachment=True,
                        download_name=f'scraping_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{format}')
    except Exception as e:
        logger.error(f"Error downloading results: {e}")
        return jsonify({'error': str(e)}), 500

# Run the app using create_app
def create_app():
    return app

# Run the app if executed directly
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
    except Exception as e:
        logger.critical(f"Failed to start application: {e}")
        sys.exit(1)

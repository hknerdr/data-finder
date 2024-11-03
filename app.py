# File: app.py

import os
import sys
import json
import csv
import re
import time
import random
import logging
import threading
import requests
import phonenumbers
import pycountry
import statistics
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from flask import Flask, render_template, request, jsonify, send_file
from fake_useragent import UserAgent
from tenacity import retry, stop_after_attempt, wait_exponential
from urllib.parse import urlparse, parse_qs
from pathlib import Path

# Enhanced logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('scraper.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Update the Keywords class with more specific and varied keywords
class Keywords:
    INDUSTRY_KEYWORDS = [
        'beverage distributor', 'drink wholesaler', 'soft drinks supplier',
        'juice distributor', 'water distributor', 'beverage company',
        'drinks supplier', 'beverage wholesaler', 'soda distributor',
        'mineral water supplier'
    ]
    
    BUSINESS_KEYWORDS = [
        'contact', 'directory', 'supplier list', 'business directory',
        'company profile', 'contact details', 'email', 'phone',
        'address', 'wholesale contact'
    ]

class ContactExtractor:
    EMAIL_REGEX = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    PHONE_REGEX = r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    
    @staticmethod
    def extract_emails(text: str) -> List[str]:
        emails = re.findall(ContactExtractor.EMAIL_REGEX, text)
        return list(set([email.lower() for email in emails if '@' in email]))
    
    @staticmethod
    def extract_phones(text: str, country_code: str) -> List[str]:
        phones = []
        matches = re.findall(ContactExtractor.PHONE_REGEX, text)
        for match in matches:
            try:
                number = phonenumbers.parse(match, country_code)
                if phonenumbers.is_valid_number(number):
                    formatted = phonenumbers.format_number(
                        number, 
                        phonenumbers.PhoneNumberFormat.INTERNATIONAL
                    )
                    phones.append(formatted)
            except Exception:
                continue
        return list(set(phones))

class WebScraper:
    def __init__(self, proxy_manager):
        self.session = requests.Session()
        self.ua = UserAgent()
        self.proxy_manager = proxy_manager
        
    def get_headers(self) -> Dict[str, str]:
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def scrape_url(self, url: str, country_code: str) -> Dict[str, List[str]]:
        try:
            headers = self.get_headers()
            proxy = self.proxy_manager.get_proxy()
            response = self.session.get(
                url,
                headers=headers,
                proxies=proxy.to_dict() if proxy else None,
                timeout=15,
                verify=False
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            text_content = soup.get_text()
            
            emails = ContactExtractor.extract_emails(text_content)
            phones = ContactExtractor.extract_phones(text_content, country_code)
            
            # Also check for contact page links
            contact_links = soup.find_all('a', href=re.compile(r'contact|about|get-in-touch', re.I))
            for link in contact_links:
                href = link.get('href')
                if href:
                    if not href.startswith(('http://', 'https://')):
                        href = f"{urlparse(url).scheme}://{urlparse(url).netloc}{href}"
                    try:
                        contact_response = self.session.get(
                            href,
                            headers=headers,
                            proxies=proxy.to_dict() if proxy else None,
                            timeout=10
                        )
                        if contact_response.status_code == 200:
                            contact_soup = BeautifulSoup(contact_response.text, 'html.parser')
                            contact_text = contact_soup.get_text()
                            emails.extend(ContactExtractor.extract_emails(contact_text))
                            phones.extend(ContactExtractor.extract_phones(contact_text, country_code))
                    except Exception as e:
                        logger.warning(f"Failed to scrape contact page {href}: {e}")
            
            return {
                'emails': list(set(emails)),
                'phones': list(set(phones))
            }
            
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return {'emails': [], 'phones': []}

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
            logger.debug("Monitoring session started.")

    def end_session(self):
        with self.lock:
            self.metrics.end_time = datetime.now()
            logger.debug("Monitoring session ended.")

    def record_request(self, success: bool):
        with self.lock:
            self.metrics.total_requests += 1
            if not success:
                self.metrics.failed_requests += 1
            logger.debug(f"Recorded request: success={success}")

    def record_extraction(self, success: bool):
        with self.lock:
            if success:
                self.metrics.successful_extractions += 1
            else:
                self.metrics.failed_extractions += 1
            logger.debug(f"Recorded extraction: success={success}")

    def log_error(self, error: str, context: str = None):
        with self.lock:
            self.error_log.append({
                'timestamp': datetime.now().isoformat(),
                'error': str(error),
                'context': context
            })
            logger.error(f"Logged error: {error} | Context: {context}")

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
            logger.debug(f"Recorded request time: {duration} seconds")

    def record_extraction_time(self, duration: float):
        with self.lock:
            self.extraction_times.append(duration)
            logger.debug(f"Recorded extraction time: {duration} seconds")

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
        logger.error(f"Recorded error of type '{error_type}': {error}")

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
            logger.debug(f"Updated metrics: {new_metrics}")

class SearchEngine:
    def __init__(self, state: 'GlobalState'):
        self.state = state
        self.session = requests.Session()
        self.user_agent = UserAgent()
        self.search_delay = random.uniform(10, 15)  # Increased delay between searches
        
    def search(self, query: str) -> List[str]:
    try:
        logger.info(f"Starting search for query: {query}")
        headers = {
            'User-Agent': self.user_agent.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Use a single search domain initially for testing
        search_url = 'https://www.google.com/search'
        
        params = {
            'q': query,
            'hl': 'en',
            'num': 10,  # Reduced number for testing
            'start': 0
        }
        
        logger.debug(f"Making request to {search_url} with params: {params}")
        
        # Add timeout to prevent hanging
        response = self.session.get(
            search_url,
            params=params,
            headers=headers,
            proxies=self.get_proxies(),
            timeout=30,
            verify=False
        )
        
        if response.status_code == 200:
            logger.info(f"Successful response received for query: {query}")
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Store the results
            results = []
            
            # Try multiple selector patterns
            selectors = [
                'div.g div.yuRUbf > a[href]',
                'div.rc > div.r > a[href]',
                'div.g a[href]'
            ]
            
            for selector in selectors:
                links = soup.select(selector)
                if links:
                    logger.debug(f"Found {len(links)} links using selector: {selector}")
                    for link in links:
                        url = link.get('href')
                        if url and url.startswith('http') and 'google.' not in url:
                            results.append(url)
                            logger.debug(f"Added URL to results: {url}")
            
            # Fallback method if no results found
            if not results:
                logger.debug("No results found with primary selectors, trying fallback method")
                urls = re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s<>"\']*', response.text)
                results = [url for url in urls if 'google.' not in url]
            
            unique_results = list(set(results))
            logger.info(f"Found {len(unique_results)} unique URLs for query: {query}")
            return unique_results
            
        else:
            logger.error(f"Search failed with status code: {response.status_code}")
            return []
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for query '{query}': {str(e)}")
        self.state.monitoring_system.log_error(str(e), context=f"Search query: {query}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in search for query '{query}': {str(e)}")
        self.state.monitoring_system.log_error(str(e), context=f"Search query: {query}")
        return []

    def get_proxies(self) -> Optional[Dict[str, str]]:
        if not self.state.proxies:
            return None
        proxy = random.choice(self.state.proxies)
        logger.debug(f"Using proxy: {proxy.get_url()}")
        return proxy.to_dict()

# Proxy Validation Function
def validate_proxy(proxy: Proxy) -> bool:
    try:
        proxies = proxy.to_dict()
        logger.debug(f"Validating proxy: {proxy.get_url()}")
        response = requests.get("https://httpbin.org/ip", proxies=proxies, timeout=10)
        is_valid = response.status_code == 200
        logger.debug(f"Proxy {proxy.get_url()} validation result: {is_valid}")
        return is_valid
    except Exception as e:
        logger.error(f"Proxy validation failed for {proxy.get_url()}: {e}")
        return False

# Update the scraping worker function to handle rate limiting
def scraping_worker(country: str, keywords: List[str] = None):
    max_retries = 3
    retry_count = 0
    
    try:
        queries = generate_search_queries(country, keywords)
        total_queries = len(queries)
        
        with global_state.lock:
            global_state.scraping_status['total'] = total_queries
        
        global_state.monitoring_system.start_session()
        logger.info(f"Starting scraping for country: {country} with {total_queries} queries")
        
        request_manager = RequestManager()
        
        for idx, query in enumerate(queries, 1):
            with global_state.lock:
                if not global_state.scraping_status['is_running']:
                    break
            
            while retry_count < max_retries:
                try:
                    time.sleep(random.uniform(15, 30))
                    
                    status_message = f"Processing query {idx}/{total_queries}: {query} (Attempt {retry_count + 1})"
                    logger.info(status_message)
                    
                    with global_state.lock:
                        global_state.scraping_status['current_status'] = status_message
                        global_state.scraping_status['progress'] = idx
                    
                    results = global_state.search_engine.search(query)
                    
                    if results:
                        logger.info(f"Found {len(results)} results for query: {query}")
                        break
                    else:
                        retry_count += 1
                        logger.warning(f"No results found for query: {query}, attempt {retry_count}")
                        time.sleep(random.uniform(30, 60))
                        
                except Exception as e:
                    retry_count += 1
                    logger.error(f"Error processing query '{query}': {e}")
                    with global_state.lock:
                        global_state.scraping_status['errors'].append(str(e))
                    global_state.monitoring_system.log_error(str(e), context=f"Query: {query}")
                    time.sleep(random.uniform(30, 60))
            
            retry_count = 0  # Reset for next query
            
            # Process results if any were found
            if results:
                process_results(results, request_manager, country)
                
    except Exception as e:
        logger.error(f"Critical error in scraping worker: {e}")
        with global_state.lock:
            global_state.scraping_status['is_running'] = False
            global_state.scraping_status['errors'].append(str(e))
        global_state.monitoring_system.log_error(str(e), context="Scraping worker critical error")
    finally:
        global_state.monitoring_system.end_session()
        logger.info("Scraping worker finished execution")

def process_results(results: List[str], request_manager: RequestManager, country: str):
    """Process the found URLs and extract contact information."""
    for url in results:
        with global_state.lock:
            if url in global_state.data_manager.processed_urls:
                continue
            global_state.data_manager.processed_urls.add(url)
        
        try:
            time.sleep(random.uniform(5, 10))
            contacts = extract_contacts_from_url(url, request_manager, country_code=country)
            
            if contacts['emails'] or contacts['phones']:
                with global_state.lock:
                    global_state.scraping_status['results'].append({
                        'URL': url,
                        'Emails': contacts['emails'],
                        'Phones': contacts['phones']
                    })
                global_state.monitoring_system.record_extraction(True)
                logger.info(f"Successfully extracted contacts from {url}")
        except Exception as e:
            logger.error(f"Failed to process URL {url}: {e}")
            global_state.monitoring_system.record_extraction(False)

# Utility Functions
def generate_search_queries(country: str, keywords: List[str] = None) -> List[str]:
    if not keywords:
        keywords = [f"{industry} {business}" for industry in Keywords.INDUSTRY_KEYWORDS for business in Keywords.BUSINESS_KEYWORDS]
    # Use country name instead of code
    country_name = get_country_name(country)
    queries = [f"{keyword} {country_name}" for keyword in keywords]
    logger.debug(f"Generated {len(queries)} search queries.")
    return queries

def get_country_name(country_code: str) -> str:
    try:
        country = pycountry.countries.get(alpha_2=country_code.upper())
        if country:
            return country.name
        else:
            logger.warning(f"Invalid country code: {country_code}. Using as is.")
            return country_code
    except Exception as e:
        logger.error(f"Error getting country name for code '{country_code}': {e}")
        return country_code

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
            logger.debug(f"Extracted {len(emails)} emails and {len(phones)} phones from {url}")
    except Exception as e:
        logger.error(f"Failed to extract contacts from {url}: {e}")
        global_state.monitoring_system.log_error(e, context=f"URL: {url}")
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
            logger.debug("Rotated User-Agent.")
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
            logger.debug(f"Making request to {url} with headers {headers} and proxies {proxies}")
            start_time = time.time()
            response = self.session.request(method, url, **kwargs)
            duration = time.time() - start_time
            global_state.performance_optimizer.record_request_time(duration)
            response.raise_for_status()
            logger.debug(f"Received response from {url} in {duration:.2f} seconds.")
            logger.debug(f"Response content snippet: {response.text[:500]}")  # İlk 500 karakteri logla
            global_state.monitoring_system.record_request(True)
            return response
        except Exception as e:
            logger.error(f"Request failed for {url}: {e}")
            global_state.monitoring_system.record_request(False)
            global_state.monitoring_system.log_error(e, context=f"URL: {url}")
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
        logger.debug(f"Removed duplicates. Cleaned results count: {len(cleaned)}")
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
            logger.debug(f"Added proxy: {proxy.get_url()}")
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
            logger.debug(f"Proxy {proxy.get_url()} validation: {is_valid}")
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
            if global_state.scraping_status['is_running']:
                return jsonify({'error': 'Scraping is already running'}), 400
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
            logger.info(f"Scraping started with country: {country} and keywords: {keywords}")

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
            if not global_state.scraping_status['is_running']:
                return jsonify({'message': 'Scraping is not running'}), 200
            global_state.scraping_status['is_running'] = False
            logger.info("Scraping has been requested to stop.")
        return jsonify({'message': 'Scraping stopped'}), 200
    except Exception as e:
        logger.error(f"Error stopping scraping: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def get_status():
    try:
        with global_state.lock:
            status_copy = global_state.scraping_status.copy()
        metrics = global_state.monitoring_system.get_metrics()
        performance = global_state.performance_optimizer.get_performance_metrics()
        errors = global_state.error_tracker.get_error_summary()
        response = {
            **status_copy,
            'metrics': metrics,
            'performance': performance,
            'errors': errors
        }
        logger.debug(f"Status requested: {response}")
        return jsonify(response)
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
        logger.info(f"Providing download for {filename}")
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

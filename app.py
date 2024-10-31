# Add these imports to the existing ones
import json
from pathlib import Path
from typing import Dict, List, Optional
import urllib3
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class ProxyConfig:
    protocol: str
    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None
    
    def get_url(self) -> str:
        auth = f"{self.username}:{self.password}@" if self.username and self.password else ""
        return f"{self.protocol}://{auth}{self.host}:{self.port}"
    
    def to_dict(self) -> Dict[str, str]:
        url = self.get_url()
        return {
            'http': url,
            'https': url
        }

class ProxyConfigManager:
    def __init__(self, config_path: str = "config/proxy_config.json"):
        self.config_path = Path(config_path)
        self.proxies: List[ProxyConfig] = []
        self.settings: Dict = {}
        self.last_load_time = None
        self.load_config()
    
    def load_config(self) -> None:
        try:
            if not self.config_path.exists():
                self.create_default_config()
            
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            self.proxies = [
                ProxyConfig(**proxy_data)
                for proxy_data in config_data.get('proxies', [])
            ]
            self.settings = config_data.get('settings', {})
            self.last_load_time = datetime.now()
            
            logging.info(f"Loaded {len(self.proxies)} proxies from configuration")
        
        except Exception as e:
            logging.error(f"Error loading proxy configuration: {str(e)}")
            self.proxies = []
            self.settings = {}
    
    def create_default_config(self) -> None:
        default_config = {
            "proxies": [],
            "settings": {
                "rotation_interval": 5,
                "max_failures": 3,
                "timeout": 30
            }
        }
        
        self.config_path.parent.mkdir(exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
    
    def get_all_proxies(self) -> List[Dict[str, str]]:
        if not self.proxies:
            return []
        
        return [proxy.to_dict() for proxy in self.proxies]
    
    def validate_proxy(self, proxy: ProxyConfig) -> bool:
        try:
            test_url = "https://www.google.com"
            with requests.Session() as session:
                response = session.get(
                    test_url,
                    proxies=proxy.to_dict(),
                    timeout=self.settings.get('timeout', 30)
                )
                return response.status_code == 200
        except Exception as e:
            logging.warning(f"Proxy validation failed for {proxy.host}: {str(e)}")
            return False

class EnhancedProxyManager(ProxyManager):
    def __init__(self):
        super().__init__()
        self.config_manager = ProxyConfigManager()
        self.proxy_failures: Dict[str, int] = {}
        self.last_rotation = datetime.now()
    
    def get_proxy(self) -> Optional[Dict[str, str]]:
        # Reload config periodically
        if (datetime.now() - self.config_manager.last_load_time) > timedelta(minutes=5):
            self.config_manager.load_config()
        
        # Check if it's time to rotate
        if (datetime.now() - self.last_rotation).seconds > self.config_manager.settings.get('rotation_interval', 300):
            self.current_proxy = None
            self.last_rotation = datetime.now()
        
        if not self.current_proxy:
            available_proxies = [
                proxy for proxy in self.config_manager.get_all_proxies()
                if self.proxy_failures.get(str(proxy), 0) < self.config_manager.settings.get('max_failures', 3)
            ]
            
            if available_proxies:
                self.current_proxy = random.choice(available_proxies)
            else:
                self.proxy_failures.clear()
                return None
        
        return self.current_proxy
    
    def mark_proxy_failed(self, proxy: Dict[str, str]) -> None:
        proxy_key = str(proxy)
        self.proxy_failures[proxy_key] = self.proxy_failures.get(proxy_key, 0) + 1
        if self.proxy_failures[proxy_key] >= self.config_manager.settings.get('max_failures', 3):
            logging.warning(f"Proxy {proxy_key} exceeded maximum failures")
            self.current_proxy = None

# Update the RequestManager to use EnhancedProxyManager
class RequestManager:
    def __init__(self):
        self.ua = UserAgent()
        self.proxy_manager = EnhancedProxyManager()
        self.session = requests.Session()
        self.request_count = 0
    
    def get_headers(self) -> Dict[str, str]:
        self.request_count += 1
        if self.request_count % Config.USER_AGENT_REFRESH_RATE == 0:
            self.ua = UserAgent()
        
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
        }
    
    @retry(
        stop=stop_after_attempt(Config.RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def make_request(self, url: str, method: str = 'get', **kwargs) -> requests.Response:
        try:
            proxy = self.proxy_manager.get_proxy()
            kwargs.update({
                'headers': self.get_headers(),
                'timeout': Config.REQUEST_TIMEOUT,
                'proxies': proxy,
                'verify': True
            })
            
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            
            # Add delay to respect rate limits
            time.sleep(Config.RATE_LIMIT_DELAY)
            
            return response
            
        except Exception as e:
            if proxy:
                self.proxy_manager.mark_proxy_failed(proxy)
            raise

def extract_phones(text: str, country_code: str) -> List[str]:
    """Enhanced phone number extraction and validation"""
    phones = set()
    
    # Common phone patterns
    patterns = [
        r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]',
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        r'\b\d{10}\b',
        r'\+\d{1,3}\s?\d{10}\b'
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            phone = match.group()
            clean_phone = re.sub(r'[\s\(\)\-\.]', '', phone)
            
            # Add country code if missing
            if not clean_phone.startswith('+'):
                clean_phone = f"+{country_code}{clean_phone}"
            
            try:
                parsed_number = phonenumbers.parse(clean_phone)
                if phonenumbers.is_valid_number(parsed_number):
                    formatted_number = phonenumbers.format_number(
                        parsed_number,
                        phonenumbers.PhoneNumberFormat.INTERNATIONAL
                    )
                    phones.add(formatted_number)
            except phonenumbers.NumberParseException:
                continue
    
    return list(phones)
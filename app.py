from flask import Flask, render_template, request, jsonify, send_file
import requests
from bs4 import BeautifulSoup
import pandas as pd
import threading
import time
import re
import phonenumbers
import logging
from urllib.parse import urlparse
import datetime
import nltk
from nltk.tokenize import sent_tokenize
import json
import io
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Download necessary NLTK data
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logging.warning(f"NLTK data download failed. Some features might be limited. Error: {str(e)}")

# Global variables to store scraping state
scraping_status = {
    'is_running': False,
    'progress': 0,
    'total': 0,
    'current_status': '',
    'results': []
}

def is_valid_email(email):
    regex = r'^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w+$'
    return re.match(regex, email, re.IGNORECASE) is not None

def is_valid_phone(phone, country_code):
    try:
        parsed_number = phonenumbers.parse(phone, country_code.upper())
        return phonenumbers.is_valid_number(parsed_number)
    except phonenumbers.NumberParseException:
        return False

def extract_address(text):
    address_patterns = [
        r'\d+\s+[A-Za-z0-9\s,.-]+(?:Road|Rd|Street|St|Avenue|Ave|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b',
        r'P\.?O\.?\s*Box\s+\d+',
        r'[A-Za-z0-9\s]+(?:Business Park|Industrial Estate|Technology Park|Office Park|Center|Centre)',
        r'[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?'
    ]
    
    for pattern in address_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return max(matches, key=len)
    
    try:
        sentences = sent_tokenize(text)
        for sentence in sentences:
            if any(word.isdigit() for word in sentence.split()) and \
               any(word in sentence for word in ['Street', 'Road', 'Avenue', 'Suite', 'Floor']):
                return sentence.strip()
    except Exception:
        pass
    
    return 'N/A'

def extract_contact_details(url, country_code):
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; ContactInfoBot/1.0)',
        'Accept': 'text/html,application/xhtml+xml'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title/name
        name = soup.title.string if soup.title else 'N/A'
        
        # Extract emails
        emails = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', response.text)
        email = next((e for e in emails if is_valid_email(e)), 'N/A')
        
        # Extract phones
        phones = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', response.text)
        phone = 'N/A'
        for p in phones:
            clean_phone = re.sub(r'[\s\(\)\-\.]', '', p)
            if not clean_phone.startswith('+'):
                clean_phone = '+' + clean_phone
            if is_valid_phone(clean_phone, country_code):
                phone = clean_phone
                break
        
        # Extract address
        address = extract_address(response.text)
        
        return {
            'Name': name,
            'Email': email,
            'Phone': phone,
            'Address': address,
            'Website': url
        }
    except Exception as e:
        logging.error(f"Failed to extract details from {url}: {e}")
        return None

def google_search(query, start_index, api_key, search_engine_id):
    url = 'https://www.googleapis.com/customsearch/v1'
    params = {
        'key': api_key,
        'cx': search_engine_id,
        'q': query,
        'start': start_index
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Google search failed: {e}")
        return None

def scraping_worker(api_key, search_engine_id, country, keywords):
    global scraping_status
    scraping_status['results'] = []
    scraping_status['is_running'] = True
    scraping_status['progress'] = 0
    
    try:
        total_keywords = len(keywords)
        scraping_status['total'] = total_keywords * 10  # Approximate number of results
        
        for keyword in keywords:
            if not scraping_status['is_running']:
                break
                
            query = f"{keyword} in {country}"
            scraping_status['current_status'] = f"Searching for: {query}"
            
            for start_index in range(1, 11):  # Get first 10 pages
                if not scraping_status['is_running']:
                    break
                    
                search_results = google_search(query, start_index, api_key, search_engine_id)
                if not search_results or 'items' not in search_results:
                    continue
                
                for item in search_results['items']:
                    if not scraping_status['is_running']:
                        break
                        
                    url = item.get('link')
                    if url:
                        details = extract_contact_details(url, country)
                        if details:
                            scraping_status['results'].append(details)
                            
                    scraping_status['progress'] += 1
                    
                time.sleep(1)  # Be nice to the servers
                
    except Exception as e:
        logging.error(f"Scraping error: {e}")
    finally:
        scraping_status['is_running'] = False
        scraping_status['current_status'] = 'Completed'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start_scraping', methods=['POST'])
def start_scraping():
    data = request.json
    api_key = data.get('api_key')
    search_engine_id = data.get('search_engine_id')
    country = data.get('country')
    keywords = data.get('keywords', '').split('\n')
    
    if not all([api_key, search_engine_id, country, keywords]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    # Stop any existing scraping
    global scraping_status
    scraping_status['is_running'] = False
    time.sleep(1)  # Wait for previous thread to stop
    
    # Start new scraping thread
    thread = threading.Thread(
        target=scraping_worker,
        args=(api_key, search_engine_id, country, keywords)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Scraping started'})

@app.route('/stop_scraping', methods=['POST'])
def stop_scraping():
    global scraping_status
    scraping_status['is_running'] = False
    return jsonify({'message': 'Scraping stopped'})

@app.route('/status')
def get_status():
    return jsonify(scraping_status)

@app.route('/download')
def download_results():
    if not scraping_status['results']:
        return jsonify({'error': 'No results available'}), 404
    
    df = pd.DataFrame(scraping_status['results'])
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name='scraping_results.csv'
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
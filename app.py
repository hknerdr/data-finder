import streamlit as st
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

# Configure logging
logging.basicConfig(filename='scraper.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Global variables
stop_scraping = False

# Function to log messages
def log(message):
    logging.info(message)
    st.write(message)

# Function to validate email addresses
def is_valid_email(email):
    regex = r'^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w+$'
    return re.match(regex, email, re.IGNORECASE) is not None

# Function to validate phone numbers
def is_valid_phone(phone, country_code):
    try:
        parsed_number = phonenumbers.parse(phone, country_code)
        return phonenumbers.is_valid_number(parsed_number)
    except phonenumbers.NumberParseException:
        return False

# Function to check robots.txt
def can_crawl(url, user_agent='*'):
    parsed_url = urlparse(url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
    try:
        response = requests.get(robots_url, timeout=5)
        if response.status_code == 200:
            robots_txt = response.text
            lines = robots_txt.split('\n')
            allow = True  # Assume allowed unless disallowed
            for line in lines:
                if line.strip().lower().startswith('user-agent'):
                    ua = line.split(':', 1)[1].strip()
                    if ua == '*' or ua == user_agent:
                        allow = True
                    else:
                        allow = False
                elif line.strip().lower().startswith('disallow') and allow:
                    disallowed_path = line.split(':', 1)[1].strip()
                    if parsed_url.path.startswith(disallowed_path):
                        return False
            return True
        else:
            return True  # No robots.txt, proceed cautiously
    except Exception:
        return True  # If robots.txt can't be reached, proceed cautiously

# Function to extract contact details from a website
def extract_contact_details(url, country_code):
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; YourScriptName/1.0)'}
    try:
        if not can_crawl(url):
            log(f"Skipping {url} due to robots.txt restrictions.")
            return None

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Initialize variables
        name = soup.find('title').get_text(strip=True) if soup.find('title') else 'N/A'
        email = 'N/A'
        phone = 'N/A'
        address = 'N/A'

        # Search for email addresses
        emails_found = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', soup.get_text())
        for email_candidate in emails_found:
            if is_valid_email(email_candidate):
                email = email_candidate
                break

        # Search for phone numbers
        phones_found = re.findall(r'\+?\d[\d\s\-().]{7,}', soup.get_text())
        for phone_candidate in phones_found:
            phone_clean = re.sub(r'[^\d+]', '', phone_candidate)
            if is_valid_phone(phone_clean, country_code):
                phone = phone_clean
                break

        # Address extraction is complex; requires advanced parsing
        # For simplicity, we can look for common address patterns or keywords
        # Here we leave it as 'N/A' unless specific patterns are identified

        return {
            'Name': name,
            'Email': email,
            'Phone': phone,
            'Address': address,
            'Website': url
        }
    except Exception as e:
        log(f"Failed to extract details from {url}: {e}")
        return None

# Function to perform Google Custom Search
def google_search(query, start_index, api_key, search_engine_id):
    url = 'https://www.googleapis.com/customsearch/v1'
    params = {
        'key': api_key,
        'cx': search_engine_id,
        'q': query,
        'start': start_index  # Start index for pagination
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        log(f"Failed to retrieve search results from Google: {e}")
        return None

# Main scraping function
def main(api_key, search_engine_id, country, keywords_list):
    global stop_scraping
    stop_scraping = False
    all_companies = []
    total_keywords = len(keywords_list)
    companies_processed = 0
    total_companies_estimate = total_keywords * 100  # Adjust as needed

    # Initialize progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()

    for keyword in keywords_list:
        if stop_scraping:
            break
        log(f"Searching for keyword: {keyword}")
        for start_index in range(1, 101, 10):  # Pagination
            if stop_scraping:
                break
            query = f"{keyword} in {country}"
            search_results = google_search(query, start_index, api_key, search_engine_id)
            if not search_results:
                break
            web_pages = search_results.get('items', [])
            if not web_pages:
                break
            company_urls = [page.get('link') for page in web_pages if page.get('link')]

            for company_url in company_urls:
                if stop_scraping:
                    break
                log(f"Visiting: {company_url}")
                company_details = extract_contact_details(company_url, country_code=country)
                if company_details:
                    all_companies.append(company_details)
                    # Save incrementally
                    df = pd.DataFrame(all_companies)
                    df.drop_duplicates(subset=['Website'], inplace=True)
                    df.to_csv('beverage_companies.csv', index=False)
                companies_processed += 1

                # Update progress
                elapsed_time = time.time() - start_time
                if companies_processed > 0:
                    time_per_company = elapsed_time / companies_processed
                    companies_remaining = total_companies_estimate - companies_processed
                    estimated_time_remaining = companies_remaining * time_per_company
                    eta = datetime.timedelta(seconds=int(estimated_time_remaining))
                    progress = companies_processed / total_companies_estimate
                    progress_bar.progress(min(progress, 1.0))
                    status_text.text(f"Processed {companies_processed}/{total_companies_estimate} companies, ETA: {eta}")
                time.sleep(2)  # Polite delay

    if all_companies:
        df = pd.DataFrame(all_companies)
        df.drop_duplicates(subset=['Website'], inplace=True)
        df.to_csv('beverage_companies_final.csv', index=False)
        df.to_json('beverage_companies.json', orient='records', lines=True)
        log("Data collection complete. Results saved to 'beverage_companies_final.csv' and 'beverage_companies.json'.")
        st.success("Scraping completed successfully.")
        # Provide a download link
        st.markdown(get_table_download_link(df), unsafe_allow_html=True)
    else:
        log("No data collected.")
        st.warning("No data collected.")

# Function to provide a download link for the DataFrame
def get_table_download_link(df):
    import base64
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="beverage_companies.csv">Download CSV File</a>'
    return href

# Streamlit UI
def run_app():
    st.title("Beverage Distributors Scraper")

    st.markdown("""
    This application allows you to search for beverage distributors, wholesalers, and traders in a specified country.

    **Instructions:**
    - Enter the country name.
    - Modify the default keywords if necessary.
    - Enter your Google Custom Search API key and Search Engine ID.
    - Click **Start Scraping** to begin.
    """)

    # Input fields
    country = st.text_input("Country:")

    # Default keywords
    default_keywords = """
Non-Alcoholic Beverage Distributor
Soft Drink Wholesaler
Juice Supplier
Energy Drink Trader
Bottled Water Dealer
Functional Beverage Reseller
Herbal Drink Importer
Smoothie Stockist
Plant-Based Beverage Merchant
Kombucha Exporter
""".strip()

    keywords_input = st.text_area("Keywords (one per line):", value=default_keywords, height=200)

    api_key = st.text_input("Google API Key:", type="password")
    search_engine_id = st.text_input("Search Engine ID (cx):")

    # Start and Stop buttons
    start_button = st.button("Start Scraping")
    stop_button = st.button("Stop Scraping")

    if start_button:
        if not country or not keywords_input or not api_key or not search_engine_id:
            st.error("Please fill in all fields.")
        else:
            keywords_list = [k.strip() for k in keywords_input.split('\n') if k.strip()]
            # Start scraping in a new thread
            threading.Thread(target=main, args=(api_key, search_engine_id, country, keywords_list)).start()

    if stop_button:
        global stop_scraping
        stop_scraping = True
        log("Scraping process stopped by user.")
        st.warning("Scraping process stopped by user.")

if __name__ == "__main__":
    run_app()

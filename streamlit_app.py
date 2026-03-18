import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from google import genai  # Your preferred import

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="FacultyScan", layout="wide", page_icon="🧬")
ORCID_CLIENT_ID = 'APP-2LRNHFE202EQNXMJ'
ORCID_CLIENT_SECRET = 'ed968a2b-ef1a-40a4-a3a2-cf10fcbac017'
FILE_PATH = "data/pubs2.csv"

# --- 2. DATA ENGINES ---

@st.cache_data
def load_local_data(path):
    try:
        df = pd.read_csv(path)
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
        df['ORCID'] = df['ORCID'].astype(str).str.strip()
        return df
    except Exception:
        return None

@st.cache_data
def get_single_faculty_info(orcid_id, client_id, client_secret):
    try:
        auth_res = requests.post("https://orcid.org/oauth/token", data={
            "client_id": client_id, "client_secret": client_secret,
            "grant_type": "client_credentials", "scope": "/read-public"
        }).json()
        token = auth_res.get("access_token")
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        
        p_res = requests.get(f"https://pub.orcid.org/v3.0/{orcid_id}/person", headers=headers).json()
        first = p_res.get('name', {}).get('given-names', {}).get('value', '')
        last = p_res.get('name', {}).get('family-name', {}).get('value', '')
        
        e_res = requests.get(f"https://pub.orcid.org/v3.0/{orcid_id}/employments", headers=headers).json()
        org = "Independent"
        affiliations = e_res.get('affiliation-group', [])
        if affiliations:
            summ = affiliations[0].get('summaries', [{}])[0].get('employment-summary', {})
            org = summ.get('organization', {}).get('name', 'Unknown')
            
        return {"name": f"{first} {last}".strip(), "org": org, "first": first, "last": last}
    except Exception:
        return None

@st.cache_data
def scrape_jhm_by_url(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200: return "Profile missing.", []
        soup = BeautifulSoup(res.text, "html.parser")
        
        # 1. Biography
        bio_div = soup.find('div', {'id': 'overview'}) or \
                  soup.find('div', {'data-testid': 'profile-about-about_the_provider-content'})
        bio_txt = bio_div.get_text(separator=' ', strip=True) if bio_div else "Bio details missing."
        
        # 2. Education & Training
        edu_items = []
        seen = set()
        keywords = ["Fellowship", "Residency", "Medical Education", "Board Certifications", "Internship"]
        
        # We iterate through all <li> and <div> tags to find the labels
        for item in soup.find_all(['li', 'div', 'p']):
            text_content = item.get_text(separator=' ', strip=True)
            
            for key in keywords:
                if text_content.startswith(key) and key not in seen:
                    # Clean the string: remove the keyword, colon, and extra whitespace
                    # This automatically ignores the <span> tags and just gets the text
                    clean_value = text_content.replace(key, "").strip(": ").strip()
                    
                    # Prevent grabbing massive blocks (stop at the next category if merged)
                    for k in keywords:
                        clean_value = clean_value.split(k)[0].strip()
                    
                    if len(clean_value) > 2:
                        edu_items.append(f"{key}: {clean_value}")
                        seen.add(key) # Use key here so we only get ONE of each category
                            
        return bio_txt, edu_items
    except Exception: 
        return "Error loading profile.", []

# --- 3. MAIN EXECUTION ---
df = load_local_data(FILE_PATH)

if df is not None:
    # --- 4. SIDEBAR ---
    with st.sidebar:
        st.header("Settings")
        gemini_key = st.text_input("Gemini API Key", type="password", key="gemini_key_input")
        
        st.divider()
        st.header("Researcher Selection")
        all_orcids = sorted(df['ORCID'].unique().tolist())
        selected_orcid = st.selectbox("Select ORCID", options=all_orcids, key="orcid_selector")
        
        st.divider()
        year_range = st.slider("Year Range", int(df['Year'].min()), 2026, (2010, 2026))
        search_query = st.text_input("Filter Titles", "").lower()

    faculty = get_single_faculty_info(selected_orcid, ORCID_CLIENT_ID, ORCID_CLIENT_SECRET)

    if faculty:
        # --- 5. TOP LEVEL INFO ---
        st.title(f"Faculty Profile: {faculty['name']}")
        
        # Row 1: Organization
        st.metric("Organization", faculty['org'])
        
        # Row 2: ORCID ID
        st.metric("ORCID ID", selected_orcid)
        
        st.divider()

        # Scrape Info
        f_slug = faculty['first'].lower().replace(" ", "-")
        l_slug = faculty['last'].lower().replace(" ", "-")
        jhu_url = f"https://profiles.hopkinsmedicine.org/provider/{f_slug}-{l_slug}/2703089"
        bio, edu_list = scrape_jhm_by_url(jhu_url)

        # --- 6. BIO & EDUCATION (Sequential Rows) ---
        st.markdown(f"🔗 **Source Profile:** [{jhu_url}]({jhu_url})")
        
        st.subheader("Biography")
        st.write(bio)
        
        st.write("") 
        st.subheader("Education & Training")
        if edu_list:
            for item in edu_list:
                st.markdown(f"• **{item}**")
        else:
            st.info("Detailed education records not found on the profile.")
        st.divider()

        # --- 7. PUBLICATIONS ---
        mask = (df['ORCID'] == str(selected_orcid)) & \
               (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])
        
        if search_query:
            mask &= df['Title'].str.contains(search_query, case=False, na=False)
            
        pub_df = df[mask].copy().sort_values("Year", ascending=False)
        pub_df.insert(0, "No.", range(1, len(pub_df) + 1))

        st.subheader(f"Publications ({len(pub_df)})")
        st.dataframe(
            pub_df[["No.", "Year", "Title", "Abstract", "PubMed Link"]],
            column_config={
                "No.": st.column_config.Column(width="small"),
                "Year": st.column_config.NumberColumn(format="%d", width="small"),
                "Title": st.column_config.Column(width=800), # Or a pixel value like 400
                "Abstract": st.column_config.Column(width=1600), # Or 600+
                "PubMed Link": st.column_config.LinkColumn("Link"),
            },
            hide_index=True, use_container_width=True
        )

        # --- 8. GEMINI AI SECTION ---
        st.divider()
        st.header(f"🤖 AI Research Insights")
        
        if not gemini_key:
            st.info("👈 Please enter your Gemini API key in the sidebar.")
        else:
            try:
                client = genai.Client(api_key=gemini_key)
                model_id = "gemini-3.1-flash-lite-preview" # Updated to flash for speed

                # Prepare context
                context_parts = [f"Name: {faculty['name']}", f"Bio: {bio}", f"Training: {', '.join(edu_list)}"]
                for _, row in pub_df.iterrows():
                    context_parts.append(f"({row['Year']}) {row['Title']}: {row.get('Abstract', 'N/A')}")
                full_history = "\n\n".join(context_parts)
                
                # Summary with Session Caching
                if "gemini_summary" not in st.session_state or st.session_state.get('cur_orcid') != selected_orcid:
                    with st.spinner("Gemini is analyzing the full history..."):
                        prompt = f"Provide a professional 4-5 sentence summary of {faculty['name']}'s research and background. Context: {full_history}"
                        response = client.models.generate_content(model=model_id, contents=prompt)
                        st.session_state.gemini_summary = response.text
                        st.session_state.cur_orcid = selected_orcid
                
                st.info(st.session_state.gemini_summary)
                
                # Chat Interface
                st.write("---")
                if "gemini_chat" not in st.session_state: st.session_state.gemini_chat = []
                
                for msg in st.session_state.gemini_chat:
                    with st.chat_message(msg["role"]): st.markdown(msg["content"])

                if user_q := st.chat_input("Ask a question about this researcher..."):
                    st.session_state.gemini_chat.append({"role": "user", "content": user_q})
                    with st.chat_message("user"): st.markdown(user_q)
                    
                    with st.chat_message("assistant"):
                        chat_prompt = f"Context: {full_history}\n\nQuestion: {user_q}"
                        chat_res = client.models.generate_content(model=model_id, contents=chat_prompt)
                        st.markdown(chat_res.text)
                        st.session_state.gemini_chat.append({"role": "assistant", "content": chat_res.text})
            
            except Exception as e:
                st.error(f"Gemini AI Error: {e}")
else:
    st.error("Data file (pubs2.csv) not found.")
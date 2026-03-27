import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from google import genai 

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="FacultyScan Multi-Compare", layout="wide", page_icon="🧬")
ORCID_CLIENT_ID = 'APP-VYOKD26NG7YD3EPW'
ORCID_CLIENT_SECRET = '51ffc00e-d65c-4e64-8073-cefb9357a813'
PUBS_FILE = "data/pubs4.csv"
PROFILES_FILE = "data/profiles.csv"

# --- 2. DATA ENGINES ---

@st.cache_data
def load_data():
    try:
        df_pubs = pd.read_csv(PUBS_FILE)
        df_pubs['Year'] = pd.to_numeric(df_pubs['Year'], errors='coerce').fillna(0).astype(int)
        df_pubs['ORCID'] = df_pubs['ORCID'].astype(str).str.strip()
        for col in ['Authors', 'Abstract']:
            if col in df_pubs.columns:
                df_pubs[col] = df_pubs[col].astype(str).replace('nan', 'N/A')
            else:
                df_pubs[col] = "N/A"
        
        df_profiles = pd.read_csv(PROFILES_FILE)
        df_profiles['ORCID_ID'] = df_profiles['ORCID_ID'].astype(str).str.strip()
        df_profiles['Name'] = df_profiles['Name'].astype(str).str.strip()
        
        url_map = dict(zip(df_profiles['ORCID_ID'], df_profiles['Hopkins_Profile_URL']))
        name_to_orcid = {f"{row['Name']} ({row['ORCID_ID']})": row['ORCID_ID'] 
                         for _, row in df_profiles.iterrows()}
        
        return df_pubs, url_map, name_to_orcid
    except Exception as e:
        st.error(f"Error loading CSV files: {e}")
        return None, None, None

@st.cache_data
def get_orcid_token(client_id, client_secret):
    try:
        res = requests.post("https://orcid.org/oauth/token", data={
            "client_id": client_id, "client_secret": client_secret,
            "grant_type": "client_credentials", "scope": "/read-public"
        })
        return res.json().get("access_token")
    except: return None

@st.cache_data
def get_full_faculty_details(orcid_id, token):
    try:
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
        return {"name": f"{first} {last}".strip(), "org": org, "first": first, "last": last, "orcid": orcid_id}
    except: return None

@st.cache_data
def scrape_jhm_by_url(url):
    if not url or pd.isna(url): return "No profile URL provided.", []
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200: return "Profile missing.", []
        soup = BeautifulSoup(res.text, "html.parser")
        bio_div = soup.find('div', {'id': 'overview'}) or \
                  soup.find('div', {'data-testid': 'profile-about-about_the_provider-content'})
        bio_txt = bio_div.get_text(separator=' ', strip=True) if bio_div else "Bio details missing."
        edu_items = []
        seen = set()
        keywords = ["Fellowship", "Residency", "Medical Education", "Board Certifications", "Internship"]
        for item in soup.find_all(['li', 'div', 'p']):
            text_content = item.get_text(separator=' ', strip=True)
            for key in keywords:
                if text_content.startswith(key) and key not in seen:
                    clean_val = text_content.replace(key, "").strip(": ").strip()
                    for k in keywords: clean_val = clean_val.split(k)[0].strip()
                    if len(clean_val) > 2:
                        edu_items.append(f"{key}: {clean_val}")
                        seen.add(key)
        return bio_txt, edu_items
    except: return "Error loading profile.", []

def generate_with_fallback(client, prompt, model_list):
    for model_id in model_list:
        try:
            response = client.models.generate_content(model=model_id, contents=prompt)
            return response, model_id
        except: continue
    return None, None

# --- 3. MAIN EXECUTION ---
df_pubs, url_map, name_to_orcid = load_data()

if df_pubs is not None:
    token = get_orcid_token(ORCID_CLIENT_ID, ORCID_CLIENT_SECRET)
    
    with st.sidebar:
        st.header("Settings")
        gemini_key = st.text_input("Gemini API Key", type="password")
        st.divider()
        st.header("Researcher Selection")
        selected_display_names = st.multiselect("Select Researchers", options=sorted(list(name_to_orcid.keys())))
        selected_orcids = [name_to_orcid[name] for name in selected_display_names]
        
        current_year = datetime.now().year
        five_years_ago = current_year - 4 
        st.divider()
        year_range = st.slider("Year Range", min_value=int(df_pubs['Year'].min()), 
                               max_value=current_year, value=(five_years_ago, current_year))
        search_query = st.text_input("Filter Titles/Abstracts", "").lower()

    if selected_orcids:
        all_ai_context = [] 

        for orcid in selected_orcids:
            faculty = get_full_faculty_details(orcid, token)
            if faculty:
                with st.expander(f"👤 {faculty['name']} - {faculty['org']}", expanded=True):
                    col1, col2 = st.columns([1, 1])
                    with col1: st.metric("Organization", faculty['org'])
                    with col2: st.metric("ORCID ID", orcid)

                    jhu_url = url_map.get(orcid)
                    bio, edu_list = scrape_jhm_by_url(jhu_url)
                    if jhu_url: st.markdown(f"🔗 **Source Profile:** [{jhu_url}]({jhu_url})")
                    
                    st.subheader("Biography")
                    st.write(bio)
                    
                    mask = (df_pubs['ORCID'] == str(orcid)) & \
                           (df_pubs['Year'] >= year_range[0]) & (df_pubs['Year'] <= year_range[1])
                    if search_query:
                        mask &= (df_pubs['Title'].str.contains(search_query, case=False, na=False)) | \
                                (df_pubs['Abstract'].str.contains(search_query, case=False, na=False))
                        
                    current_pub_df = df_pubs[mask].copy().sort_values("Year", ascending=False)
                    current_pub_df.insert(0, "No.", range(1, len(current_pub_df) + 1))

                    st.subheader(f"Publications ({len(current_pub_df)})")
                    st.dataframe(
                        current_pub_df[["No.", "Year", "Title", "Authors", "Abstract", "PubMed Link"]],
                        column_config={
                            "PubMed Link": st.column_config.LinkColumn("Link"),
                            "Abstract": st.column_config.Column(width="large"),
                        },
                        hide_index=True, use_container_width=True
                    )

                    # --- REVISED Context for AI ---
                    researcher_context = f"RESEARCHER: {faculty['name']}\n"
                    researcher_context += f"Bio: {bio}\n"
                    researcher_context += f"Training: {', '.join(edu_list) if edu_list else 'N/A'}\n"
                    researcher_context += "Key Publications:\n"
                    
                    for _, row in current_pub_df.head(8).iterrows():
                        researcher_context += f"- YEAR: {row['Year']}\n"
                        researcher_context += f"  TITLE: {row['Title']}\n"
                        researcher_context += f"  AUTHORS: {row['Authors']}\n"  # Added Author List
                        researcher_context += f"  ABSTRACT SAMPLE: {row['Abstract'][:400]}...\n\n"
                    
                    all_ai_context.append(researcher_context)
                    st.divider()
        
        # --- 6. AI SECTION ---
        st.header(f"🤖 AI Research Insights & Chat")
        if not gemini_key:
            st.info("👈 Please enter your Gemini API key in the sidebar.")
        elif all_ai_context:
            try:
                client = genai.Client(api_key=gemini_key)
                models_to_try = ["gemini-flash-latest", "gemini-flash-lite-latest", "gemini-2.5-flash", "gemini-2.0-flash"]
                full_history = "\n\n---\n\n".join(all_ai_context)
                
                # --- INITIAL SUMMARY ---
                current_selection_key = "-".join(selected_orcids)
                if "gemini_summary" not in st.session_state or st.session_state.get('cur_selection') != current_selection_key:
                    with st.spinner("Gemini is analyzing the group..."):
                        prompt = f"Summarize these researchers and shared themes based on this data: {full_history}"
                        response, success_model = generate_with_fallback(client, prompt, models_to_try)
                        if response:
                            st.session_state.gemini_summary = response.text
                            st.session_state.active_model = success_model
                            st.session_state.cur_selection = current_selection_key
                
                if "gemini_summary" in st.session_state:
                    st.caption(f"Status: Connected via **{st.session_state.get('active_model')}**")
                    st.info(st.session_state.gemini_summary)

                # --- CHAT INTERFACE (RESTORED) ---
                st.write("### Chat with AI about these researchers")
                if "group_chat" not in st.session_state:
                    st.session_state.group_chat = []

                # Display chat history
                for msg in st.session_state.group_chat:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                # Handle new user input
                if user_q := st.chat_input("Ask a follow-up question..."):
                    st.session_state.group_chat.append({"role": "user", "content": user_q})
                    with st.chat_message("user"):
                        st.markdown(user_q)
                    
                    with st.chat_message("assistant"):
                        chat_prompt = f"Context of Researchers:\n{full_history}\n\nQuestion: {user_q}"
                        res, chat_model = generate_with_fallback(client, chat_prompt, models_to_try)
                        if res:
                            st.markdown(res.text)
                            st.session_state.group_chat.append({"role": "assistant", "content": res.text})
                        else:
                            st.error("Model unavailable for chat.")
                
            except Exception as e:
                st.error(f"Gemini API Error: {e}")
else:
    st.error("Data files not found.")
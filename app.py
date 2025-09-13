# app.py
# Streamlit helpdesk demo: classification + RAG (retriever + generator)
# Beginner-friendly single-file app. Run with: streamlit run app.py

import os
import json
import time
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

import streamlit as st
import numpy as np
from sklearn.neighbors import NearestNeighbors

# NLP / embeddings
import openai

# Utilities
import nltk
for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg)

from nltk.tokenize import sent_tokenize


# -------------------------
# Config / Defaults
# -------------------------
st.set_page_config(page_title="Customer Support Copilot (Demo)", layout="wide")

# Default Atlan docs to use for RAG (the app fetches these on first run)
DEFAULT_KB_URLS = [
"https://docs.atlan.com/get-started/how-tos/quick-start-for-admins",
    "https://docs.atlan.com/secure-agent",
    "https://docs.atlan.com/product/capabilities/playbooks",
    "https://docs.atlan.com/product/capabilities/discovery",
    "https://docs.atlan.com/product/capabilities/governance/contracts",
    "https://docs.atlan.com/product/integrations",
    "https://docs.atlan.com/product/integrations/automation",
    "https://developer.atlan.com/getting-started/",
    "https://developer.atlan.com/sdks/",
    "https://solutions.atlan.com/overview/",
    "https://docs.atlan.com/apps/connectors/data-warehouses/snowflake/how-tos/set-up-snowflake",
    "https://university.atlan.com/certifications",
    "https://security.atlan.com/",
    "https://docs.atlan.com/support/references/customer-support",
    "https://docs.atlan.com/product/capabilities/governance/access-control",
    "https://docs.atlan.com/product/integrations/identity-management/sso/troubleshooting/troubleshooting-connector-specific-sso-authentication",
    "https://developer.atlan.com/",
    "https://developer.atlan.com/snippets/",
    "https://developer.atlan.com/patterns/",
    "https://developer.atlan.com/snippets/access/tokens/",
    "https://developer.atlan.com/reference/",
    "https://developer.atlan.com/search/queries/text/",
    "https://developer.atlan.com/reference/specs/openlineage/",
    "https://developer.atlan.com/reference/specs/datacontracts/"
]

# Sample tickets (embedded from your uploaded sample file)
# Source: Sample tickets.pdf you uploaded. :contentReference[oaicite:3]{index=3}
SAMPLE_TICKETS = [

  {
    "id": "TICKET-245",
    "subject": "Connecting Snowflake to Atlan - required permissions?",
    "body": "Hi team, we're trying to set up our primary Snowflake production database as a new source in Atlan, but the connection keeps failing. We've tried using our standard service account, but it's not working. Our entire BI team is blocked on this integration for a major upcoming project, so it's quite urgent. Could you please provide a definitive list of the exact permissions and credentials needed on the Snowflake side to get this working? Thanks."
  },
  {
    "id": "TICKET-246",
    "subject": "Which connectors automatically capture lineage?",
    "body": "Hello, I'm new to Atlan and trying to understand the lineage capabilities. The documentation mentions automatic lineage, but it's not clear which of our connectors (we use Fivetran, dbt, and Tableau) support this out-of-the-box. We need to present a clear picture of our data flow to leadership next week. Can you explain how lineage capture differs for these tools?"
  },
  {
    "id": "TICKET-247",
    "subject": "Deployment of Atlan agent for private data lake",
    "body": "Our primary data lake is hosted on-premise within a secure VPC and is not exposed to the internet. We understand we need to use the Atlan agent for this, but the setup instructions are a bit confusing for our security team. This is a critical source for us, and we can't proceed with our rollout until we get this connected. Can you provide a detailed deployment guide or connect us with a technical expert?"
  },
  {
    "id": "TICKET-248",
    "subject": "How to surface sample rows and schema changes?",
    "body": "Hi, we've successfully connected our Redshift cluster, and the assets are showing up. However, my data analysts are asking how they can see sample data or recent schema changes directly within Atlan without having to go back to Redshift. Is this feature available? I feel like I'm missing something obvious."
  },
  {
    "id": "TICKET-249",
    "subject": "Exporting lineage view for a specific table",
    "body": "For our quarterly audit, I need to provide a complete upstream and downstream lineage diagram for our core `fact_orders` table. I can see the lineage perfectly in the UI, but I can't find an option to export this view as an image or PDF. This is a hard requirement from our compliance team and the deadline is approaching fast. Please help!"
  },
  {
    "id": "TICKET-250",
    "subject": "Importing lineage from Airflow jobs",
    "body": "We run hundreds of ETL jobs in Airflow, and we need to see that lineage reflected in Atlan. I've read that Atlan can integrate with Airflow, but how do we configure it to correctly map our DAGs to the specific datasets they are transforming? The current documentation is a bit high-level."
  },
  {
    "id": "TICKET-251",
    "subject": "Using the Visual Query Builder",
    "body": "I'm a business analyst and not very comfortable with writing complex SQL. I was excited to see the Visual Query Builder in Atlan, but I'm having trouble figuring out how to join multiple tables and save my query for later use. Is there a tutorial or a quick guide you can point me to?"
  },
  {
    "id": "TICKET-252",
    "subject": "Programmatic extraction of lineage",
    "body": "Our internal data science team wants to build a custom application that analyzes metadata propagation delays. To do this, we need to programmatically extract lineage data from Atlan via an API. Does the API expose lineage information, and if so, could you provide an example of the endpoint and the structure of the response?"
  },
  {
    "id": "TICKET-253",
    "subject": "Upstream lineage to Snowflake view not working",
    "body": "This is infuriating. We have a critical Snowflake view, `finance.daily_revenue`, that is built from three upstream tables. Atlan is correctly showing the downstream dependencies, but the upstream lineage is completely missing. This makes the view untrustworthy for our analysts. We've re-run the crawler multiple times. What could be causing this? This is a huge problem for us."
  },
  {
    "id": "TICKET-254",
    "subject": "How to create a business glossary and link terms in bulk?",
    "body": "We are migrating our existing business glossary from a spreadsheet into Atlan. We have over 500 terms. Manually creating each one and linking them to thousands of assets seems impossible. Is there a bulk import feature using CSV or an API to create terms and link them to assets? This is blocking our entire governance initiative."
  },
  {
    "id": "TICKET-255",
    "subject": "Creating a custom role for data stewards",
    "body": "I'm trying to set up a custom role for our data stewards. They need permission to edit descriptions and link glossary terms, but they should NOT have permission to run queries or change connection settings. I'm looking at the default roles, but none of them fit perfectly. How can I create a new role with this specific set of permissions?"
  },
  {
    "id": "TICKET-256",
    "subject": "Mapping Active Directory groups to Atlan teams",
    "body": "Our company policy requires us to manage all user access through Active Directory groups. We need to map our existing AD groups (e.g., 'data-analyst-finance', 'data-engineer-core') to teams within Atlan to automatically grant the correct permissions. I can't find the settings for this. How is this configured?"
  },
  {
    "id": "TICKET-257",
    "subject": "RBAC for assets vs. glossaries",
    "body": "I need clarification on how Atlan's role-based access control works. If a user is denied access to a specific Snowflake schema, can they still see the glossary terms that are linked to the tables in that schema? I need to ensure our PII governance is airtight."
  },
  {
    "id": "TICKET-258",
    "subject": "Process for onboarding asset owners",
    "body": "We've started identifying owners for our key data assets. What is the recommended workflow in Atlan to assign these owners and automatically notify them? We want to make sure they are aware of their responsibilities without us having to send manual emails for every assignment."
  },
  {
    "id": "TICKET-259",
    "subject": "How does Atlan surface sensitive fields like PII?",
    "body": "Our security team is evaluating Atlan and their main question is around PII and sensitive data. How does Atlan automatically identify fields containing PII? What are our options to apply tags or masks to these fields once they are identified to prevent unauthorized access?"
  },
  {
    "id": "TICKET-260",
    "subject": "Authentication methods for APIs and SDKs",
    "body": "We are planning to build several automations using the Atlan API and Python SDK. What authentication methods are supported? Is it just API keys, or can we use something like OAuth? We have a strict policy that requires key rotation every 90 days, so we need to understand how to manage this programmatically."
  },
  {
    "id": "TICKET-261",
    "subject": "Enabling and testing SAML SSO",
    "body": "We are ready to enable SAML SSO with our Okta instance. However, we are very concerned about disrupting our active users if the configuration is wrong. Is there a way to test the SSO configuration for a specific user or group before we enable it for the entire workspace?"
  },
  {
    "id": "TICKET-262",
    "subject": "SSO login not assigning user to correct group",
    "body": "I've just had a new user, 'test.user@company.com', log in via our newly configured SSO. They were authenticated successfully, but they were not added to the 'Data Analysts' group as expected based on our SAML assertions. This is preventing them from accessing any assets. What could be the reason for this mis-assignment?"
  },
  {
    "id": "TICKET-263",
    "subject": "Integration with existing DLP or secrets manager",
    "body": "Does Atlan have the capability to integrate with third-party tools like a DLP (Data Loss Prevention) solution or a secrets manager like HashiCorp Vault? We need to ensure that connection credentials and sensitive metadata classifications are handled by our central security systems."
  },
  {
    "id": "TICKET-264",
    "subject": "Accessing audit logs for compliance reviews",
    "body": "Our compliance team needs to perform a quarterly review of all activities within Atlan. They need to know who accessed what data, who made permission changes, etc. Where can we find these audit logs, and is there a way to export them or pull them via an API for our records?"
  },
  {
    "id": "TICKET-265",
    "subject": "How to programmatically create an asset using the REST API?",
    "body": "I'm trying to create a new custom asset (a 'Report') using the REST API, but my requests keep failing with a 400 error. The API documentation is a bit sparse on the required payload structure for creating new entities. Could you provide a basic cURL or Python `requests` example of what a successful request body should look like?"
  },
  {
    "id": "TICKET-266",
    "subject": "SDK availability and Python example",
    "body": "I'm a data engineer and prefer using SDKs over raw API calls. Which languages do you provide SDKs for? I'm particularly interested in Python. Where can I find the installation instructions (e.g., PyPI package name) and a short code snippet for a common task, like creating a new glossary term?"
  },
  {
    "id": "TICKET-267",
    "subject": "How do webhooks work in Atlan?",
    "body": "I'm exploring using webhooks to send real-time notifications from Atlan to our internal Slack channel. I need to understand what types of events (e.g., asset updated, term created) can trigger a webhook. Also, how do we validate that the incoming payloads are genuinely from Atlan? Do you support payload signing?"
  },
  {
    "id": "TICKET-268",
    "subject": "Triggering an AWS Lambda from Atlan events",
    "body": "We have a workflow where we want to trigger a custom AWS Lambda function whenever a specific Atlan tag (e.g., 'PII-Confirmed') is added to an asset. What is the recommended and most secure way to set this up? Should we use webhooks pointing to an API Gateway, or is there a more direct integration?"
  },
  {
    "id": "TICKET-269",
    "subject": "When to use Atlan automations vs. external services?",
    "body": "I see that Atlan has a built-in 'Automations' feature. I'm trying to decide if I should use this to manage a workflow or if I should use an external service like Zapier or our own Airflow instance. Could you provide some guidance or examples on what types of workflows are best suited for the native automations versus an external tool?"
  },
  {
    "id": "TICKET-270",
    "subject": "Connector failed to crawl - where to check logs?",
    "body": "URGENT: Our nightly Snowflake crawler failed last night and no new metadata was ingested. This is a critical failure as our morning reports are now missing lineage information. Where can I find the detailed error logs for the crawler run to understand what went wrong? I need to fix this ASAP."
  },
  {
    "id": "TICKET-271",
    "subject": "Asset extracted but not published to Atlan",
    "body": "This is very strange. I'm looking at the crawler logs, and I can see that the asset 'schema.my_table' was successfully extracted from the source. However, when I search for this table in the Atlan UI, it doesn't appear. It seems like it's getting stuck somewhere between extraction and publishing. Can you please investigate the root cause?"
  },
  {
    "id": "TICKET-272",
    "subject": "How to measure adoption and generate reports?",
    "body": "My manager is asking for metrics on our Atlan usage to justify the investment. I need to generate a report showing things like the number of active users, most frequently queried tables, and the number of assets with assigned owners. Does Atlan have a reporting or dashboarding feature for this?"
  },
  {
    "id": "TICKET-273",
    "subject": "Best practices for catalog hygiene",
    "body": "We've been using Atlan for six months, and our catalog is already starting to get a bit messy with duplicate assets and stale metadata from old tests. As we roll this out to more teams, what are some common best practices or features within Atlan that can help us maintain good catalog hygiene and prevent this problem from getting worse?"
  },
  {
    "id": "TICKET-274",
    "subject": "How to scale Atlan across multiple business units?",
    "body": "We are planning a global rollout of Atlan to multiple business units, each with its own data sources and governance teams. We're looking for advice on the best way to structure our Atlan instance. Should we use separate workspaces, or can we achieve isolation using teams and permissions within a single workspace while maintaining a consistent governance model?"
  }
]

 # ... (you can paste more tickets from the PDF if desired)


# -------------------------
# Helper functions
# -------------------------

def has_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))

def openai_client_setup():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    openai.api_key = key
    return openai

# Fallback rule-based classifier (if no OpenAI key)
def heuristic_classify(text: str) -> Dict[str, Any]:
    txt = text.lower()
    tags = []
    if any(k in txt for k in ["how to", "how do", "how can i", "how-to", "tutorial", "step"]):
        tags.append("How-to")
    if any(k in txt for k in ["snowflake", "redshift", "bigquery", "connector", "connect"]):
        tags.append("Connector")
    if any(k in txt for k in ["api", "sdk", "rest", "curl", "python", "endpoint"]):
        tags.append("API/SDK")
    if any(k in txt for k in ["sso", "okta", "saml", "login"]):
        tags.append("SSO")
    if any(k in txt for k in ["lineage", "upstream", "downstream"]):
        tags.append("Lineage")
    if not tags:
        tags = ["Product"]

    # sentiment heuristics
    sentiment = "Neutral"
    if any(k in txt for k in ["urgent", "blocked", "infuriating", "critical", "angry"]):
        sentiment = "Frustrated"
    if any(k in txt for k in ["please", "could you", "help"]):
        if sentiment == "Neutral":
            sentiment = "Curious"

    # priority heuristics
    priority = "P2"
    if any(k in txt for k in ["urgent", "critical", "blocked", "asap", "immediately"]):
        priority = "P0"
    elif any(k in txt for k in ["soon", "next week", "deadline", "important"]):
        priority = "P1"

    return {"topic_tags": tags, "sentiment": sentiment, "priority": priority}

# Use OpenAI Chat API to classify into tags/sentiment/priority
def classify_with_openai(ticket_text: str) -> Dict[str, Any]:
    system = {
        "role": "system",
        "content": (
            "You are a support triage assistant. Given a ticket subject+body, "
            "return JSON only with fields: topic_tags (list of tags), sentiment (one word), priority (one of P0,P1,P2). "
            "Topic tags should be chosen from this set: How-to, Product, Connector, Lineage, API/SDK, SSO, Glossary, Best practices, Sensitive data."
        )
    }
    user_prompt = (
        "Ticket text:\n\n"
        f"{ticket_text}\n\n"
        "Return only JSON with keys: topic_tags, sentiment, priority. Example:\n"
        '{"topic_tags":["Connector"], "sentiment":"Frustrated", "priority":"P0"}'
    )
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[system, {"role": "user", "content": user_prompt}],
        temperature=0.0,
        max_tokens=200
    )
    txt = resp["choices"][0]["message"]["content"].strip()
    try:
        parsed = json.loads(txt)
        return parsed
    except Exception as e:
        # fallback to heuristic if parse fails
        return heuristic_classify(ticket_text)

# -------------------------
# RAG: fetch, chunk, embed, store
# -------------------------

def fetch_page_text(url: str) -> str:
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        # remove scripts/styles
        for s in soup(["script", "style", "nav", "footer", "header"]):
            s.decompose()
        texts = [p.get_text(separator=" ", strip=True) for p in soup.find_all(["p", "li", "h1", "h2", "h3", "pre"])]
        return "\n".join(t for t in texts if t)
    except Exception as e:
        return ""

def chunk_text(text: str, max_chars: int = 800):
    # chunk by sentences approx up to max_chars
    sents = sent_tokenize(text)
    chunks = []
    cur = ""
    for s in sents:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = cur + " " + s if cur else s
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    # Uses OpenAI embeddings (fallback: raise if no key)
    if not has_openai_key():
        raise RuntimeError("Embeddings require OPENAI_API_KEY in environment.")
    # OpenAI embeddings API; model name may vary. Using text-embedding-3-small for cost.
    EMB_MODEL = "text-embedding-3-small"
    out = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = openai.Embedding.create(input=batch, model=EMB_MODEL)
        for item in resp["data"]:
            out.append(item["embedding"])
    return out

class SimpleVectorStore:
    def __init__(self):
        self.vectors = None  # np.array
        self.metadatas = []
        self.nn = None

    def build(self, embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        if not embeddings:
            self.vectors = np.zeros((0,1))
            self.metadatas = []
            self.nn = None
            return
        self.vectors = np.array(embeddings)
        self.metadatas = metadatas
        # fit neighbors (cosine)
        self.nn = NearestNeighbors(n_neighbors=min(5, len(self.vectors)), metric='cosine')
        self.nn.fit(self.vectors)

    def query(self, emb: List[float], top_k: int = 4):
        if self.nn is None:
            return []
        distances, indices = self.nn.kneighbors([emb], n_neighbors=min(top_k, len(self.vectors)))
        results = []
        for d, idx in zip(distances[0], indices[0]):
            md = self.metadatas[idx].copy()
            md["score"] = float(1 - d)  # convert cosine distance -> rough similarity
            results.append(md)
        return results

# Build KB from DEFAULT_KB_URLS (cache in memory)
@st.cache_data(show_spinner=False)
def build_kb(urls: List[str]):
    pages = []
    for u in urls:
        txt = fetch_page_text(u)
        if not txt:
            continue
        chunks = chunk_text(txt)
        for c in chunks:
            pages.append({"text": c, "source": u})
    if not pages:
        return {"store": SimpleVectorStore(), "pages": []}
    texts = [p["text"] for p in pages]
    metadatas = [{"source": p["source"], "text_snippet": p["text"][:500]} for p in pages]
    embeddings = embed_texts(texts)
    store = SimpleVectorStore()
    store.build(embeddings, metadatas)
    return {"store": store, "pages": pages}

# RAG generation: retrieve and call LLM to answer using retrieved chunks
def rag_answer(question: str, kb_store: SimpleVectorStore, top_k: int = 4) -> Dict[str, Any]:
    if not has_openai_key():
        return {
            "answer": "RAG requires an OPENAI_API_KEY to run the embedding + generation pipeline. Falling back to a canned reply.",
            "sources": []
        }
    # embed question
    emb = embed_texts([question])[0]
    retrieved = kb_store.query(emb, top_k=top_k)
    # build prompt
    context_parts = []
    sources = []
    for r in retrieved:
        snippet = r.get("text_snippet", "")[:800]
        src = r.get("source")
        sources.append(src)
        context_parts.append(f"Source: {src}\n---\n{snippet}\n")
    context = "\n\n".join(context_parts)
    system = {
        "role": "system",
        "content": "You are an assistant that answers support questions using only the provided context. If the answer can be found in the context, answer concisely and include a 'SOURCES' section at the end listing the URLs used. If not found, say you couldn't find the answer in the provided docs."
    }
    user = {
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    }
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[system, user],
        temperature=0.0,
        max_tokens=500
    )
    ans = resp["choices"][0]["message"]["content"].strip()
    unique_sources = list(dict.fromkeys(sources))
    return {"answer": ans, "sources": unique_sources}

# -------------------------
# Streamlit UI
# -------------------------

st.title(" Atlan Customer Support Copilot ")

st.markdown(
    "This demo ingests sample tickets, classifies them (Topic Tags / Sentiment / Priority), "
    "and runs a simple RAG pipeline (retrieval from Atlan docs + generative answer)."
)

# Sidebar: OpenAI key status + KB build control
with st.sidebar:
    st.header("Settings")
    api_key_present = has_openai_key()
    st.write("OpenAI API Key present:" , "✅" if api_key_present else "❌ (fallback mode)")
    st.markdown("**Knowledge base (RAG)**")
    st.write("Default Atlan doc URLs used for KB (editable below).")
    urls_input = st.text_area("KB URLs (one per line)", value="\n".join(DEFAULT_KB_URLS), height=200)
    build_kb_button = st.button("(Re)build KB from URLs")

# Build KB if key present
kb_urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
kb_data = None
if api_key_present:
    with st.spinner("Building KB from docs (this may take 20-40s on first run)..."):
        try:
            kb_data = build_kb(kb_urls)
            st.sidebar.success("KB built (in-memory).")
        except Exception as e:
            st.sidebar.error(f"Failed to build KB: {e}")
            kb_data = {"store": SimpleVectorStore(), "pages": []}
else:
    st.sidebar.info("No OpenAI key — KB build skipped (heuristic fallback).")
    kb_data = {"store": SimpleVectorStore(), "pages": []}

# Main columns
col1, col2 = st.columns((2, 3))

# COL 1: Bulk classification dashboard
with col1:
    st.subheader("Bulk ticket classification")
    st.write("Press **Classify all** to run classification (OpenAI or heuristic fallback).")
    if st.button("Classify all"):
        total = len(SAMPLE_TICKETS)
        rows = []

        # spinner + progress bar
        with st.spinner("Classifying tickets... please wait ⏳"):
            progress_bar = st.progress(0, text="Starting classification...")

            for idx, t in enumerate(SAMPLE_TICKETS, start=1):
                text = t["subject"] + "\n\n" + t["body"]

                if has_openai_key():
                    try:
                        c = classify_with_openai(text)
                    except Exception:
                        c = heuristic_classify(text)
                else:
                    c = heuristic_classify(text)

                rows.append({
                    "id": t["id"],
                    "subject": t["subject"],
                    "topic_tags": ", ".join(c.get("topic_tags", [])),
                    "sentiment": c.get("sentiment"),
                    "priority": c.get("priority")
                })

                # update progress
                pct = int((idx / total) * 100)
                progress_bar.progress(pct, text=f"Classifying tickets... {pct}%")

                # optional: tiny sleep to make progress visible
                time.sleep(0.1)

            st.session_state["bulk_rows"] = rows

        st.success("✅ Classification complete.")

        rows = st.session_state.get("bulk_rows", None)
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No classifications yet. Click 'Classify all' to run.")

# COL 2: Interactive AI agent
with col2:
    st.subheader("Interactive AI agent (submit a new ticket)")
    channel = st.selectbox("Channel", ["Email", "Live chat", "WhatsApp", "Voice"])
    subj = st.text_input("Subject", value="")
    body = st.text_area("Body", value="")
    if st.button("Submit ticket"):
        ticket_text = subj + "\n\n" + body
        # classify
        if has_openai_key():
            try:
                analysis = classify_with_openai(ticket_text)
            except Exception as e:
                analysis = heuristic_classify(ticket_text)
        else:
            analysis = heuristic_classify(ticket_text)
        st.markdown("### Internal analysis")
        st.json(analysis)
        # decide if RAG needed
        rag_topics = set(["How-to", "Product", "Best practices", "API/SDK", "SSO"])
        use_rag = any(t in rag_topics for t in analysis.get("topic_tags", []))
        st.markdown("### Final response")
        if use_rag and has_openai_key() and kb_data and kb_data.get("store") and kb_data["store"].nn is not None:
            with st.spinner("Running RAG to generate a contextual answer..."):
                res = rag_answer(ticket_text, kb_data["store"], top_k=4)
                st.markdown("**AI (RAG) answer:**")
                st.write(res["answer"])
                if res["sources"]:
                    st.markdown("**Citations used:**")
                    for s in res["sources"]:
                        st.write(s)
        elif use_rag and not has_openai_key():
            st.info("RAG requires an OpenAI key. Showing a helpful placeholder response.")
            st.write("I classify this ticket as a product/how-to question and would use the docs to construct a direct answer if an API key were configured.")
        else:
            # non-RAG topics -> routing message
            primary_tag = analysis.get("topic_tags", ["Product"])[0]
            st.write(f"This ticket has been classified as `{primary_tag}` and routed to the appropriate team for follow-up.")

# Footer: note about challenge spec (from uploaded challenge PDF)


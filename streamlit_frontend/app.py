import streamlit as st
import uuid
from services.api import PolicyBotAPI

# --- Page Config ---
st.set_page_config(page_title="PolicyBot Auditor", page_icon="🛡️", layout="wide")

# --- Session & API Setup ---
if "client_id" not in st.session_state:
    st.session_state.client_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

api = PolicyBotAPI(st.session_state.client_id)

# --- Sidebar ---
with st.sidebar:
    st.title("🗂️ Audit Workspace")
    st.caption(f"Session ID: {st.session_state.client_id}")
    st.divider()

    st.subheader("1. Upload Policies")
    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Process Documents", type="primary"):
        if uploaded_files:
            # 1. Start Upload
            upload_resp = api.upload_documents(uploaded_files)

            if "error" in upload_resp:
                st.error(f"Upload failed: {upload_resp['error']}")
            else:
                # 2. Listen for Real-time Progress
                # st.status creates a container for the logs
                with st.status("Processing documents...", expanded=True) as status_box:
                    st.write("Initiating background tasks...")

                    print(f"{api.listen_to_progress()}")
                    # Loop through WebSocket messages as they arrive

                    # why cant use '''  ''' for multiline comments

                    # In standard Python, a standalone string """ ... """
                    # is often used as a comment (or docstring). However, Streamlit works differently:

                    # If Streamlit sees a string sitting alone on a line (not assigned to a variable), it assumes you want to display it and
                    # automatically converts it to st.write().

                    # The for loop and the yield keyword work together like a relay race.

                    # 1. Frontend (app.py): The for loop asks: "Give me the next update."

                    # 2.API Service (listen_to_progress): The function runs until it hits ws.recv().

                    # 3.The Pause: ws.recv() is a blocking call. This means the entire Python script pauses execution right there. It sits and waits. It does not burn CPU; it just waits for a network packet.

                    # 4. The Message: When the Backend sends a message (e.g., "Parsing..."), ws.recv() wakes up.

                    # 5. The Yield: The function yields the data.

                    # 6. Frontend (app.py): The for loop receives the data, prints it to the screen, and then loops back to Step 1.

                    for update in api.listen_to_progress():
                        msg = update.get("message") or update.get("detail")
                        state = update.get("status")

                        print(f"status from app.py for loops:{state}")

                        st.write(f"⚙️ {msg}")  # Print log

                        if state == "error":
                            status_box.update(label="Processing Failed", state="error")
                            st.error(msg)

                    # Final success state
                    status_box.update(
                        label="Ingestion Complete!", state="complete", expanded=False
                    )
                    st.success("System Ready! You can now ask questions.")
        else:
            st.warning("Please select a file first.")

    st.divider()

    if st.button("🗑️ Reset Session"):
        api.reset_session()
        st.session_state.messages = []
        st.rerun()

# --- Main Chat Area ---
st.title("🛡️ PolicyBot Auditor")

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "details" in message:
            with st.expander("View Evidence"):
                st.markdown(f"**Reasoning:** {message['details']['reasoning']}")
                st.markdown(f"**Evidence:** {message['details']['evidence']}")

# Chat Input
if prompt := st.chat_input("Ask a compliance question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            response = api.ask_question(prompt)

            if "error" in response:
                st.error(response["error"])
            else:
                answer = response.get("answer", "No answer provided.")
                st.markdown(answer)

                # Save details
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "details": {
                            "reasoning": response.get("reasoning", "N/A"),
                            "evidence": response.get("evidence", "N/A"),
                        },
                    }
                )

                with st.expander("View Evidence"):
                    st.markdown(f"**Reasoning:** {response.get('reasoning')}")
                    st.markdown(f"**Evidence:** {response.get('evidence')}")

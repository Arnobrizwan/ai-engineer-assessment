# VHS tape — records docs/demo.gif (the live Q2 streaming + memory demo).
# Render with:  vhs docs/demo.tape   (requires a running Q2 server on :8000)
Output docs/demo.gif

Set Shell "bash"
Set FontSize 16
Set Width 1040
Set Height 300
Set Padding 18
Set Theme "Dracula"
Set TypingSpeed 35ms

# --- hidden setup (session is reset externally before rendering) ---
Hide
Type "cd q2-streaming-chat && clear" Enter
Show

Type "# Q2 — FastAPI endpoint streaming the LLM token-by-token (SSE)" Enter
Sleep 700ms
Type "# Turn 1: introduce myself" Enter
Sleep 400ms
Type "./client.sh 'My name is Arnob. Reply with just: noted.' demo" Enter
Sleep 5s
Enter
Type "# Turn 2: a fresh request in the SAME session — does it remember?" Enter
Sleep 400ms
Type "./client.sh 'What is my name? Answer in one word.' demo" Enter
Sleep 5s
Enter
Type "# Recalled from DB-persisted history => chat session memory works." Enter
Sleep 2500ms

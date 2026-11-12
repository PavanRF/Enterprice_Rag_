# Enterprise Document Assistant (RAG with Role-Based Access Control)

A secure internal assistant that answers questions from company documents. Each person sees only the documents their role allows, the access rule is enforced before retrieval, and every change is checked by an automated security and quality gate.

## Features

- **Sign-in** with salted password hashes, lockout after repeated failures, and generic error messages.
- **Customizable look:** background presets, an optional photo background, and accent color, all set in `config/branding.yaml`. Users can switch the background on the login page and in the sidebar.
- **Ask questions** in a chat with source citations, follow-up questions that use the conversation, and suggested starting questions.
- **"What documents can I access?"** returns the real list for the user's role. Restricted documents are never listed.
- **Document library** for browsing and searching readable documents.
- **Upload and delete** documents within your department's write permission. Uploads are validated, and personal data in a file triggers a warning.
- **Feedback and export:** thumbs up or down on answers, and a Markdown export of each chat.
- **Analytics page** for executives: usage, refusals, cost, and feedback. Question text is never shown, to protect employee privacy.
- **Guardrails:** personal data masking, prompt-injection refusal, out-of-scope refusal, and grounded answers only.
- **Evaluation gate:** labelled security and quality tests that fail the build when a threshold is missed.

## Setup

```powershell
python -m pip install -r requirements.txt
python -m rag.cli set-password priya.sharma     # repeat for each user you want to sign in
python -m streamlit run app\streamlit_app.py
```

Passwords must be at least 10 characters. Sign in with a username such as `priya.sharma`, which you can find in `config/access.yaml`.

To use the Groq model, copy `.env.example` to `.env` and set `GROQ_API_KEY`. Keep the key out of code, screenshots, and chat.

## Customizing the look

Edit `config/branding.yaml`:

- `app_name`, `tagline`, and `accent_color` change the text and button color.
- `backgrounds` lists the presets users can choose. Each value is any CSS background.
- `background_image` names a file in `assets/` (for example `office.jpg`). It adds a "Photo" choice.

## Permissions

`config/access.yaml` controls who sees and changes what:

- `role_grants` sets which departments each role can **read**.
- `write_grants` sets which departments each role can **upload to and delete from**.
- `analytics_roles` sets who can open the Analytics page.
- `users` maps each person to their roles.

A document belongs to the department of the folder it is in (`data/docs/<department>/`).

## Command line

```powershell
python -m rag.cli ingest                              # show indexed documents per department
python -m rag.cli ask --user priya.sharma "What was Q3 revenue?"
python -m rag.cli eval                                # quality and security gate
python -m rag.cli stats
python -m unittest discover -s tests -v               # 84 tests
```

## How answers are produced

1. Personal data in the question is masked. Prompt-injection and out-of-scope requests are refused.
2. "What documents can I access?" returns the document list directly.
3. A follow-up question is rewritten into a standalone question using the recent conversation (requires the LLM).
4. The search only considers documents the user's role can read. Titles are indexed with their text.
5. The model answers only from those documents, with citations. If nothing supports an answer, the assistant says so.
6. Personal data in the answer is masked. The question and answer are logged without any sensitive values.

## Security notes

- Access is enforced before retrieval, so restricted text never reaches the model, the answer, or the citations.
- A refusal is identical whether a document is restricted or doesn't exist, so the assistant never reveals what you can't see.
- Passwords are stored as salted PBKDF2 hashes in `data/auth/credentials.json`, which is git-ignored.
- Upload and delete paths are validated so a crafted filename cannot leave the documents folder.
- In `AUTH_MODE=proxy`, your reverse proxy must strip any client-supplied copy of the identity header.

## Current limitations

- **Keyword retrieval.** Search uses TF-IDF with title indexing. It misses some paraphrases, which is why the evaluation shows one known miss (`mkt-03`) without the LLM.
- **Follow-up resolution and fluent answers need the Groq key.** Without it, answers are quoted sentences, and follow-up questions are not rewritten.
- **In-memory index.** It rebuilds when a document changes, which is fine for a small library, not for thousands of files.
- **Local accounts.** Production should use your company's identity provider (`AUTH_MODE=proxy`) rather than local passwords.
- **The lockout counter is per running instance.** Multiple servers need a shared store.

## Author

[Your Name] · [LinkedIn] · [GitHub] · [Email]

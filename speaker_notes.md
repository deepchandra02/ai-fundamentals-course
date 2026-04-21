# Speaker Notes — GenAI Fundamentals

> Slide-by-slide talking points for the presenter. Each section maps to one slide in `slides.md`.

---

## Slide 1: Title — GenAI Fundamentals

- Welcome everyone to Session 3 in our Python-AI series.
- In Sessions 1-2 we learned Python basics. Today we take a step back from code and build a conceptual foundation for what Generative AI actually is and how it works.
- By the end of this hour, you'll understand the core ideas behind tools like ChatGPT and Claude — and more importantly, how to extend them with your own data.
- No code to run today — this is all about building intuition. We'll get hands-on with Python in the next sessions.

---

## Slide 2: Traditional AI vs Generative AI

- Before we dive into GenAI, let's quickly frame what makes it different from the AI/ML you may have heard about for years.
- Traditional AI is about **classification and prediction** — "Is this spam?", "Will this customer churn?". It picks from known categories.
- Generative AI **creates new content** — text, images, code — that didn't exist before. That's a fundamental shift.
- The key question has changed from "what is this?" to "make me something new."
- Quick check: Has anyone here used ChatGPT, Claude, or Copilot? (Likely most hands go up — good, that's all GenAI.)

---

## Slide 3: The GenAI Landscape

- Here's a quick lay of the land — who the major players are.
- On the left: **closed-source** models. You access these via APIs. You don't see the model weights. OpenAI (GPT-4), Anthropic (Claude), Google (Gemini).
- On the right: **open-source** models. You can download these, run them on your own hardware, modify them. Meta's Llama, Mistral, and many others.
- The practical difference: closed-source is easier to start with (just call an API), open-source gives you more control and privacy.
- This landscape changes fast — new models every few months. What matters are the underlying concepts, which stay stable.

---

## Slide 4: Why Now? Three Breakthroughs

- People often ask: "AI has been around for decades — why did everything change in 2022-2023?"
- Three things converged:
  1. **Transformers** (2017) — Google's "Attention Is All You Need" paper introduced the architecture that all modern LLMs use. The key innovation is the "attention mechanism" — the model can look at the whole input at once, not just left-to-right.
  2. **Scaling laws** (~2020) — Researchers discovered that if you just make models bigger and train on more data, they keep getting better. This seems obvious now but was a major insight.
  3. **RLHF** (2022) — Reinforcement Learning from Human Feedback. This is what turned a "text prediction engine" into a helpful assistant. Humans ranked responses, and the model learned to produce what humans preferred.
- The combination of all three gave us ChatGPT in late 2022, and the field has been sprinting ever since.

---

## Slide 5: Section Title — How LLMs Work

- Now let's look under the hood. This is the longest section because it builds the foundation for everything else.
- Don't worry — we'll use analogies and visuals, not math.

---

## Slide 6: Tokens — The Atoms of Language

- This is the first surprise: LLMs don't read words like we do. They read **tokens**.
- A token is roughly ¾ of a word. Common words are one token, but longer or unusual words get split up.
- Walk through the example: "The cat sat on the mat" becomes 6 tokens, each mapped to a number.
- Show the surprising examples — "Unbelievable" becomes 4 tokens. The model doesn't see the word, it sees pieces.
- **This matters because:** Context windows, pricing, everything is measured in tokens. When someone says "128K context window" they mean 128,000 tokens, not words.

---

## Slide 7: The Prediction Machine

- Here's the core insight of the entire session: **An LLM does ONE thing — it predicts the next token.**
- That's it. All the magic — writing poems, coding, summarizing documents — comes from predicting "what token is most likely to come next?"
- Walk through the probability box: given "The cat sat on the", the model says "mat" is 32% likely, "floor" is 18%, etc.
- Then show the generation process: it predicts one token, feeds that back in, predicts the next, and so on. It's autoregressive — each step depends on the last.
- **Important to emphasize:** There's no "understanding" happening. It's incredibly sophisticated pattern matching. This helps explain both why it's so capable AND why it makes mistakes.

---

## Slide 8: Training Pipeline

- Three stages to build an LLM — each builds on the last.
- **Pre-training:** The most expensive part. The model reads billions of web pages, books, code. It learns the patterns of language, facts about the world, and how to code. This costs millions of dollars and takes months. The result is a model that can predict text, but it's not yet useful as an assistant.
- **Fine-tuning:** Take that base model and train it on Q&A pairs — "here's a question, here's a good answer." Now it learns to be helpful rather than just completing text.
- **RLHF:** Human evaluators rank multiple responses — "response A is better than B." The model learns human preferences. This is what makes it refuse harmful requests and be genuinely helpful.
- The cost column tells a story: pre-training is a massive investment (only big companies do it), fine-tuning is accessible, RLHF requires human labor.

---

## Slide 9: Context Window = Working Memory

- The context window is one of the most important practical concepts.
- Everything — the system prompt, the conversation so far, the documents you've pasted in, AND the response being generated — must fit inside this window.
- Walk through the diagram: system prompt + conversation + response = everything on the "whiteboard."
- Show the sizes table. These numbers keep growing — Claude started at 8K, now 200K. Gemini is over 1M.
- **The whiteboard analogy works well:** Imagine working at a whiteboard. You can only use what's written on it right now. If it fills up, you have to erase old stuff. The model has no separate "long-term memory" — context window is all it has.
- This directly motivates RAG later: if you need the model to use your 500-page manual, it won't fit in context. You need to be strategic about what you put on the whiteboard.

---

## Slide 10: Temperature — The Creativity Dial

- Temperature is a practical setting you'll use when building applications.
- Low temperature (0) = the model almost always picks the highest-probability token. Deterministic, repeatable, precise.
- High temperature (1.0+) = probabilities are flattened, less-likely tokens get a real chance. More creative, more varied, but also more risk of nonsense.
- Walk through the use cases table. For a customer service bot, you want low temp. For brainstorming, higher.
- **Practical tip:** If you're building anything factual or business-critical, keep temperature low. Save high temperature for creative applications.

---

## Slide 11: Section 2 Recap

- Quick pause to let everything sink in.
- Walk through each bullet as a one-sentence reminder.
- Emphasize the key insight: LLMs are pattern completion engines. This framing explains both their power and their limitations — which is exactly where we're going next.
- Ask: "Any questions before we move on?" (Good natural break point.)

---

## Slide 12: Section Title — Limitations, Risks & Realities

- So we've seen how impressive these models are. Now let's talk about where they fall short — because understanding limitations is critical before you build anything.

---

## Slide 13: The Three Technical Limitations

- Walk through each column:
  1. **Hallucinations:** The model is a prediction machine — it predicts plausible-sounding text. Sometimes plausible-sounding is completely wrong. It doesn't "know" facts, it predicts likely next tokens. That's why it can confidently say the Eiffel Tower is 500m tall — it's generating plausible text, not checking a database.
  2. **Knowledge cutoff:** The model was trained on data up to a certain date. Ask it about something after that date and it genuinely has no information. It's like asking someone who's been in a cave for a year about current events.
  3. **No private data:** This is the big one for enterprise use. The model was trained on public internet text. It has no idea what your company's refund policy is, what's in your internal wiki, or what your Q3 numbers look like.
- These three limitations together create the motivation for RAG.

---

## Slide 14: Responsible AI Considerations

- Before we solve the technical limitations, we need to talk about responsibility.
- **Bias in data:** The training data is the internet — which contains all the biases of society. Models can reflect and amplify stereotypes around gender, race, etc. This is especially dangerous in high-stakes applications like hiring or lending.
- **Data privacy:** Every time you paste text into ChatGPT or call an API, you're sending that data to a third party. Ask yourself: would I email this to a stranger? If not, think twice about sending it to an API. Some companies have strict policies about this.
- **Human oversight:** AI should assist, not decide. For any consequential decision — hiring, medical, legal — a human needs to review. "The AI said so" is not a defense.
- These three questions on screen are a good checklist for any AI deployment.

---

## Slide 15: Cost Awareness & The Knowledge Gap

- Quick practical point on costs — especially relevant since some of you will be building things.
- Walk through the pricing table. Token costs seem tiny — $2-5 per million tokens. But at scale, a busy chatbot can burn through millions of tokens daily.
- The $100+/day example is real — this is why prompt engineering and efficient design matter.
- Then transition to the gap visual: this is the core tension. LLMs know general stuff, you need them to know YOUR stuff. RAG bridges that gap — and that's what the rest of the session is about.

---

## Slide 16: Section Title — Embeddings & Vector Search

- To understand RAG, we first need to understand its building blocks: embeddings and vector search. This is where things get really interesting.

---

## Slide 17: What Are Embeddings?

- An embedding is a list of numbers that captures the **meaning** of text.
- Walk through the example: "king" and "queen" produce similar lists of numbers because they have similar meanings (royalty, power, etc.). "Banana" produces very different numbers because it means something completely different.
- The numbers aren't arbitrary — they're learned during training. Each dimension captures some aspect of meaning.
- In practice, embeddings are 768 to 1536 numbers long. We only show a few here for simplicity.
- **This is the key technology that makes semantic search possible.** Instead of searching for exact words, you search for similar meanings.

---

## Slide 18: Semantic Space — Meaning Has a Map

- This is the visual payoff. Imagine plotting every piece of text in a space where position = meaning.
- Happy/joyful/cheerful cluster together. Sad/unhappy/depressed cluster together. Programming terms cluster together. Vehicles cluster together.
- The distance between points reflects semantic similarity.
- **The practical magic:** "forgot my login" and "password reset help" are near each other in this space, even though they share zero words in common. This is why semantic search is so powerful compared to keyword search.
- This works across languages too — "chat" (French for cat) ends up near "cat" in the embedding space.

---

## Slide 19: Cosine Similarity

- So how do we actually measure "how similar" two embeddings are? Cosine similarity.
- Don't worry about the math — think of it as: two arrows pointing the same direction = similar (score near 1.0). Two arrows at right angles = unrelated (score near 0). Opposite directions = opposite meaning.
- Walk through the table — the scores are intuitive. "Reset my password" and "forgot my login credentials" score 0.92 — very similar! "Reset my password" and "what's the weather" score 0.13 — completely unrelated.
- Note the cat/feline example: 0.89 similarity despite using completely different words. That's the power of embeddings.

---

## Slide 20: Vector Databases

- Now that we have embeddings and a way to measure similarity, we need somewhere to store millions of them and search efficiently. That's a vector database.
- Walk through the side-by-side comparison. This is the "aha moment" for many people.
- **Keyword search** requires exact word matches. If someone types "I can't log in" and your docs say "password reset", keyword search finds nothing.
- **Vector search** converts both the query and documents to embeddings, then finds the closest matches by meaning. "I can't log in" finds "password reset guide" because they mean similar things.
- Briefly mention the popular options — ChromaDB is great for learning and prototyping (we may use it in future sessions), Pinecone for production.

---

## Slide 21: Section Title — Chunking & Document Processing

- Now we know how to convert text to meaning-vectors and search them. But there's a practical problem: documents are big. We need to break them up first.

---

## Slide 22: Why Chunk?

- Walk through the problem: if you embed an entire 100-page document as one vector, the embedding captures a vague average of everything in the doc. It's like summarizing a whole book in one sentence — too much detail lost.
- And practically: you can't feed a 100-page document into the LLM context window. You need to pick the _relevant parts_.
- The solution: split into chunks of ~500 tokens each. Each chunk gets its own embedding. Now you can find the specific paragraph that answers the user's question.
- Walk through the benefits: focused meaning, independent retrieval, fits in context.

---

## Slide 23: Chunking Strategies

- Three approaches, each with tradeoffs:
  1. **Fixed-size:** Simplest to implement — just split every N tokens. Problem: might split mid-sentence or mid-thought. The boundary doesn't respect meaning.
  2. **Sentence/paragraph-based:** Split at natural boundaries. Respects meaning but produces uneven chunk sizes. A short paragraph might be tiny, a long one might be too big.
  3. **Semantic chunking:** The smartest approach — detect when the _topic_ changes and split there. Uses embeddings to detect shifts. Best quality but more complex to implement.
- In practice, most people start with sentence-based chunking. It's a good balance of quality and simplicity.

---

## Slide 24: Overlap

- This is a subtle but important detail.
- Walk through the "without overlap" example: the refund policy sentence gets split right in the middle. Chunk 1 has half, Chunk 2 has half. Neither is useful on its own.
- With overlap, chunks share some content at the boundaries. Both chunks have the complete refund policy sentence.
- **Rule of thumb:** 10-20% overlap. So for 500-token chunks, you'd have 50-100 tokens of overlap.
- It uses a bit more storage but dramatically improves retrieval quality. It's almost always worth it.

---

## Slide 25: Section Title — RAG

- This is the payoff section. Everything we've learned — tokens, embeddings, vector search, chunking — comes together here.
- RAG is the single most important pattern for making LLMs useful with your own data.

---

## Slide 26: What is RAG?

- Walk through the side-by-side comparison.
- **Without RAG:** You ask about your refund policy, the LLM says "I don't know" because it was never trained on your internal docs.
- **With RAG:** Before the LLM answers, the system searches your documents, finds the relevant section, and includes it in the prompt. Now the LLM can answer accurately, citing section 4.2.
- **The open-book exam analogy** is very effective: "Would you rather take a test from memory, or be able to look things up in the textbook?" RAG is the open-book version.
- The LLM isn't "learning" your data — it's just reading the relevant passages right before answering. That's the beauty of it: no expensive retraining needed.

---

## Slide 27: The Full RAG Pipeline

- This is the centerpiece diagram — spend time here.
- **Phase 1 (Ingestion)** happens once, or whenever your docs change:
  1. Take your documents (PDFs, wikis, whatever)
  2. Chunk them into pieces
  3. Embed each chunk (convert to numbers)
  4. Store the embeddings in a vector database
- **Phase 2 (Query)** happens every time a user asks a question:
  1. Embed the question (same embedding model)
  2. Search the vector DB for similar chunks
  3. Retrieve the top 3-5 most relevant chunks
  4. Send those chunks + the question to the LLM
  5. LLM generates an answer using YOUR data as context
- Walk through both phases slowly, pointing at each box.
- Emphasize: the user doesn't see any of this. They just ask a question and get an answer grounded in your data.

---

## Slide 28: RAG in Pseudocode

- This is a preview of what the actual code looks like. We'll build this for real in Python in future sessions.
- Walk through each line briefly — the pseudocode reads almost like English.
- Point out the prompt template at the end: "Using ONLY this context, answer the question. If not in context, say I don't know." This is key — it constrains the LLM to your data and prevents hallucination.
- **Get people excited:** This is maybe 20 lines of real code. That's all it takes to build a custom Q&A system over your company's docs.

---

## Slide 29: What is Fine-tuning?

- Before we compare RAG and fine-tuning, let's make sure everyone understands what fine-tuning actually is.
- Fine-tuning = taking a pre-trained model and training it further on your specific data. The model's weights actually change — it "learns" your patterns.
- **The chef analogy works well:** Imagine hiring a professionally trained chef (that's the base model — already knows how to cook). Now you teach them your restaurant's specific recipes, plating style, and flavor profile. They don't forget how to cook — they just get specialized.
- Walk through the three use cases: custom brand voice, domain-specific language (medical/legal), consistent output format.
- **Key distinction to set up the next slide:** Fine-tuning changes the model itself. RAG doesn't — it just gives the model the right information at query time. That difference drives everything in the comparison that follows.

---

## Slide 30: RAG vs Fine-tuning

- Now that they understand both approaches, walk through the comparison.
- **Data freshness** is the killer advantage: RAG uses your current documents. Fine-tuning bakes data into model weights at training time — to update, you retrain.
- **Cost:** RAG is cheap — store embeddings, run searches. Fine-tuning requires GPUs for hours or days.
- **Hallucination:** RAG can point to sources ("this came from page 4 of your policy doc"). Fine-tuned models may still make things up.
- **The rule of thumb:** RAG for knowledge, fine-tuning for behavior/style. 90% of enterprise use cases are knowledge-based, so RAG is usually the right choice.
- You can also combine them — fine-tune for your brand voice, then RAG for your knowledge base. But start with RAG.

---

## Slide 31: Section Title — Prompt Engineering Essentials

- Last section — quick practical tips for talking to LLMs effectively. We'll go deeper in future sessions, but here are the three most important techniques.

---

## Slide 32: System Prompts

- The system prompt is your secret weapon. It's the "instruction manual" for the LLM.
- Walk through the example: setting up a customer service persona with clear rules.
- Key points: be specific, include constraints ("say I don't know if unsure"), define the tone.
- **Practical tip:** A well-written system prompt is the single highest-leverage thing you can do to improve AI application quality. It's not just "You are a helpful assistant" — be detailed about behavior, format, and boundaries.

---

## Slide 33: Few-Shot Learning

- Few-shot = teaching by example. Show the model 2-3 examples of what you want, and it generalizes the pattern.
- Walk through the example: we show two informal→formal conversions, and the model nails the third.
- This is incredibly powerful because you don't need to fine-tune or retrain anything — just put examples in the prompt.
- **Practical tip:** If the model's output isn't quite right, adding 2-3 examples of correct output is often the fastest fix.

---

## Slide 34: Chain-of-Thought

- This one's almost magical. Adding "think step by step" to a prompt dramatically improves reasoning accuracy.
- Walk through the math example. Without CoT, the model just guesses the final answer. With CoT, it shows its work — and each step being correct makes the final answer much more likely to be correct.
- This works because each step generates tokens that give the model more context for the next step. It's using its own output as "scratch paper."
- **Practical tip:** For any reasoning, comparison, or multi-step task — always ask the model to think step by step.

---

## Slide 35: Recap & What's Next

- Rapid recap — one sentence per bullet, reinforcing the key terms.
- **Important transition:** "Today was about understanding. Starting next session, we'll build. You now have the conceptual vocabulary — tokens, embeddings, chunks, RAG — to understand what the code is actually doing."
- Walk through the "coming up next" list to build excitement for future sessions.
- Open floor for questions.

---

## Slide 36: Quick Reference Card

- This is a takeaway — encourage people to screenshot or bookmark it.
- Don't read through every row. Just mention: "Here's a glossary of everything we covered. Use this as a reference when we start building in the next sessions."
- Remind them about `guide.md` — the full reference guide with diagrams they can review later.
- Thank everyone and close!

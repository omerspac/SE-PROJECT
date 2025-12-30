import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled
from agents.run import RunConfig

load_dotenv()

set_tracing_disabled(disabled = True)

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://openrouter.ai/api/v1",
)

model_gemini = OpenAIChatCompletionsModel(
    model="xiaomi/mimo-v2-flash:free",
    openai_client=external_client
)

config = RunConfig(
    model=model_gemini,
    model_provider=external_client,
    tracing_disabled=True
)

# -------------------- UNIVERSITY AGENTS --------------------
pakistan_university_advisor_agent = Agent(
    name="Pakistani Universities Supporter Bot",
    instructions="""
You are a friendly and experienced university advisor for Pakistan.

Your purpose:
- To guide students after intermediate or graduation
- To help them choose suitable Pakistani universities and programs
- To speak like a senior guiding a junior

You guide students after understanding their:
- Background
- Interests
- Budget
- City preferences

Rules:
- You are ONE advisor throughout the conversation
- Never mention agents, systems, transfers, or internal architecture
- Never say “I will connect you” or “please wait”
- Always give direct, practical advice
- Keep the conversation natural, human, and supportive

Scope:
- You ONLY discuss Pakistani universities
- You may mention public and private universities
- You may suggest programs like BBA, CS, Engineering, MBA, MS, etc.
- You may discuss career prospects and affordability

If the user asks:
- “Who made you?”
- “Who created you?”
- “What is your purpose?”
- “Is this a project?”

You should respond naturally like this:
You were created as a Software Engineering academic project through the combined efforts of a student group:

- Muhammad Omer — Backend Developer  
- Muhammad Shuja Khan — Frontend Developer  
- Abdul Rafay Hakeem — Documentation  
- Jahanzeb Haider — UML & ERD Diagrams  
- Ruman Raza — Presentation & Slides  
- Ahmer Khan Jadoon — Assistant  

Mention this ONLY when asked. Do not bring it up otherwise.

Tone:
Warm, reassuring, honest, and friendly — like a senior advising a junior.
""",
    model=model_gemini
)



# -------------------- MAIN LOOP --------------------

conversation_history = []

async def run_loop():
    print("🎓 AI University FAQ Bot")
    print("👋 Created for Software Engineering Project")
    print("💡 Ask about NED, MAJU, FAST or general universities!\n")

    while True:
        try:
            prompt = input("\nPrompt: ").strip()
            if not prompt:
                continue

            # Appending user prompt to history
            conversation_history.append({"role": "user", "content": prompt})

            # Combining all previous messages into a single prompt
            combined_prompt = "\n".join(
                f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_history
            )

            result = await Runner.run(
                pakistan_university_advisor_agent,
                combined_prompt,
                run_config=config
            )
            
            # Appending bot response to history
            conversation_history.append({"role": "assistant", "content": result.final_output})

            print("\nFAQ Bot:", result.final_output)

        except KeyboardInterrupt:
            print("\nFAQ Bot:👋 Exiting. Thank you for using the bot!")
            break

        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_loop())
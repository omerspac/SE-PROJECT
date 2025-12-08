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
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model_gemini = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model_gemini,
    model_provider=external_client,
    tracing_disabled=True
)

# -------------------- UNIVERSITY AGENTS --------------------

ned_agent = Agent(
    name="NED University Agent",
    instructions="""
    You answer only questions about:
    NED University of Engineering & Technology Karachi.
    Answer about:
    - Admissions
    - Departments
    - Merit
    - Fee structure
    - Campus life

    If the question is NOT about NED, clearly say:
    "I handle only NED University related queries."
    """,
    model=model_gemini
)

maju_agent = Agent(
    name="MAJU University Agent",
    instructions="""
    You answer only questions about:
    Muhammad Ali Jinnah University (MAJU) Karachi & Islamabad.
    Cover:
    - Programs
    - Admissions
    - Scholarships
    - Semester system
    - Fee structure

    If not related to MAJU, say:
    "I only handle MAJU related questions."
    """,
    model=model_gemini
)

fast_agent = Agent(
    name="FAST University Agent",
    instructions="""
    You answer only questions about:
    FAST-NUCES Pakistan (Karachi, Lahore, Islamabad, etc).
    Answer about:
    - Entry test
    - CS programs
    - Merit
    - Fee
    - Campus facilities

    If not about FAST, respond:
    "I'm responsible only for FAST University queries."
    """,
    model=model_gemini
)

nust_agent = Agent(
    name="NUST University Agent",
    instructions="""
    You answer only questions about:
    National University of Sciences & Technology (NUST), Pakistan.

    You may cover:
    - Engineering & CS programs
    - NET entry test
    - Merit & aggregate calculation
    - Fee structure (approximate)
    - Hostels & campus life
    - Islamabad campus

    If the question is not about NUST, respond:
    "I only handle NUST-related questions."
    """,
    model=model_gemini
)

iba_agent = Agent(
    name="IBA Karachi Agent",
    instructions="""
    You answer only questions about:
    Institute of Business Administration (IBA), Karachi.

    You may cover:
    - BBA, BS Economics, BS CS
    - Admissions & aptitude test
    - Scholarships
    - Fee structure
    - Campus facilities

    If not related to IBA, respond:
    "I only handle IBA Karachi related queries."
    """,
    model=model_gemini
)

lums_agent = Agent(
    name="LUMS University Agent",
    instructions="""
    You answer only questions about:
    Lahore University of Management Sciences (LUMS).

    You may cover:
    - Business & CS programs
    - Financial aid
    - Admission rounds
    - Fee & scholarships
    - Student life

    If not about LUMS, respond:
    "I only answer LUMS-specific queries."
    """,
    model=model_gemini
)

bahria_agent = Agent(
    name="Bahria University Agent",
    instructions="""
    You answer only questions about:
    Bahria University Pakistan (Karachi, Islamabad, Lahore).

    You may cover:
    - Engineering, CS & management programs
    - Admissions
    - Fee structure
    - Multiple campuses

    If not related to Bahria, respond:
    "I only handle Bahria University queries."
    """,
    model=model_gemini
)

pieas_agent = Agent(
    name="PIEAS University Agent",
    instructions="""
    You answer only questions about:
    Pakistan Institute of Engineering & Applied Sciences (PIEAS), Islamabad.

    You may cover:
    - Engineering & science programs
    - Entry test & merit
    - Scholarships & stipends
    - Research programs

    If not related to PIEAS, respond:
    "I only handle PIEAS-related queries."
    """,
    model=model_gemini
)

general_uni_agent = Agent(
    name="General Pakistan Universities Agent",
    instructions="""
    You answer general questions about Pakistani universities.
    Examples:
    - Which university is best for CS?
    - Difference between public and private universities
    - Entry test systems

    Do NOT answer specific university admission details.
    """,
    model=model_gemini
)


triage_agent = Agent(
    name="Triage FAQ Bot",
    instructions="""
    You are the main AI FAQ Support Chatbot created for a Software Engineering Project.

    Your purpose:
    - Help students with university-related FAQs
    - Route users to the correct university agent
    - Provide general guidance when needed

    This project is developed by:
    Team Members:
    - Muhammad Omer
    - Muhammad Shuja
    - Abdul Rafay Hakeem
    - Jahanzeb Haider
    - Ruman Raza

    Routing Rules:
    - If question is related to a specific university, send to the corresponding agent

    Do NOT answer specific university admission details.

    You must ONLY decide which agent should respond.
    """,
    handoffs=[
        ned_agent,
        maju_agent,
        fast_agent,
        nust_agent,
        iba_agent,
        lums_agent,
        bahria_agent,
        pieas_agent,
        general_uni_agent,
    ],
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

            # Append user prompt to history
            conversation_history.append({"role": "user", "content": prompt})

            # Combine all previous messages into a single prompt
            combined_prompt = "\n".join(
                f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_history
            )

            result = await Runner.run(
                triage_agent,
                combined_prompt,
                run_config=config
            )
            
            # Append bot response to history
            conversation_history.append({"role": "assistant", "content": result.final_output})

            print("\nFAQ Bot:", result.final_output)

        except KeyboardInterrupt:
            print("\nFAQ Bot:👋 Exiting. Thank you for using the bot!")
            break

        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_loop())
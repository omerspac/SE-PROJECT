import chainlit as cl
import asyncio
from main import pakistan_university_advisor_agent, config, conversation_history
from agents import Runner


@cl.on_chat_start
async def start_chat():
    await cl.Message(
        content=(
            "🎓 **AI University FAQ Bot**\n"
            "-> Created for Software Engineering Project\n"
            "-> 💡 Ask me about any Pakistani University or directions!\n"
        )
    ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    user_input = message.content.strip()

    # Adding user message to chat history
    conversation_history.append({"role": "user", "content": user_input})

    # Building combined prompt
    combined_prompt = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in conversation_history
    )

    try:
        result = await Runner.run(
            pakistan_university_advisor_agent,
            combined_prompt,
            run_config=config
        )

        bot_reply = result.final_output

        # Saving assistant message
        conversation_history.append({"role": "assistant", "content": bot_reply})

        await cl.Message(content=bot_reply).send()

    except Exception as e:
        await cl.Message(content=f"⚠️ Error: {e}").send()

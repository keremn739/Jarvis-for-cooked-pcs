from llm import ask_llm
from api.gemini import ask_gemini
from memory import (
    remember,
    get_all_memories,
    search_memories,
    forget_memory,
    get_relevant_memories,
    build_memory_prompt
)
from router import route_message
from tools import execute_tool


while True:

    message = input("You: ")

    if message.lower() in {"exit", "quit"}:
        print("JARVIS: Goodbye.")
        break


    if message.startswith("remember "):

        content = message[9:]

        try:
            result = remember(content)
        except ValueError as error:
            print(f"JARVIS: Memory was not saved: {error}")
        else:
            if result["created"]:
                print("JARVIS: Memory saved.")
            else:
                print("JARVIS: That memory is already saved.")


    elif message == "what do you remember":

        memories = get_all_memories()

        print("JARVIS remembers:")

        for memory in memories:
            print(memory[1])


    elif message.startswith("find "):

        search = message[5:]

        memories = search_memories(search)

        print("JARVIS found:")

        for memory in memories:
            print(memory[0])


    elif message == "forget everything":

        memories = get_all_memories()

        for memory in memories:
            forget_memory(memory[0])

        print("JARVIS: All memories deleted.")


    else:

        plan = route_message(message)

        for step in plan["steps"]:

            step_type = step["type"]


            if step_type == "MEMORY":

                query = step["content"] or message

                memories = get_relevant_memories(query)

                if not memories:
                    print("JARVIS: I don't have any relevant memories.")
                    continue

                prompt = build_memory_prompt(query, memories)

                print("JARVIS [Local]: ", end="")

                ask_llm(prompt)


            elif step_type == "TOOL":

                result = execute_tool(
                    step["action"],
                    step.get("target")
                )

                if result is False:

                    print("JARVIS: I can't execute that tool.")

                elif result is not None:

                    print(f"JARVIS: {result}")


            elif step_type == "LOCAL":

                query = step["content"] or message

                memories = get_relevant_memories(query)

                if memories:

                    prompt = build_memory_prompt(query, memories)

                else:

                    prompt = query


                print("JARVIS [Local]: ", end="")

                ask_llm(prompt)


            elif step_type == "CLOUD":

                query = step["content"] or message

                print("JARVIS [Cloud]: ", end="")

                print(ask_gemini(query))


            elif step_type == "BLOCKED":

                print("JARVIS: I can't help with that request.")


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
from mode import Mode, ModeState
from router import route_local, route_online
from tools import execute_tool
from interaction_history import enqueue_interaction, close_interaction_history


mode_state = ModeState()


while True:
    try:
        message = input("You: ")
    except (EOFError, KeyboardInterrupt):
        print("\nJARVIS: Goodbye.")
        close_interaction_history()
        break

    command = message.strip().lower()

    if command in {"exit", "quit"}:
        print("JARVIS: Goodbye.")
        close_interaction_history()
        break

    try:
        new_mode = mode_state.apply_command(message)
        if new_mode == Mode.LOCAL:
            print("JARVIS: Local mode is on.")

        elif new_mode == Mode.ONLINE:
            print("JARVIS: Online mode is on.")

        elif (command.startswith(("remember ", "what do you remember", "find ", "forget ")) or command == "forget everything") and mode_state.mode != Mode.LOCAL:
            print("JARVIS: Enter local mode first to use private memory.")

        elif command.startswith("remember "):
            content = message.strip()[9:]

            try:
                result = remember(content)

            except ValueError as error:
                print(f"JARVIS: Memory was not saved: {error}")

            else:
                if result["created"]:
                    print("JARVIS: Memory saved.")
                else:
                    print("JARVIS: That memory is already saved.")

        elif command == "what do you remember":

            memories = get_all_memories()

            print("JARVIS remembers:")

            for memory in memories:
                print(memory[1])

        elif command.startswith("find "):

            memories = search_memories(message.strip()[5:])

            print("JARVIS found:")

            for memory in memories:
                print(memory[0])

        elif command == "forget everything":

            memories = get_all_memories()

            for memory in memories:
                forget_memory(memory[0])

            print("JARVIS: All memories deleted.")

        else:

            plan = route_online(message) if mode_state.mode == Mode.ONLINE else route_local(message)

            for step in plan["steps"]:

                step_type = step["type"]

                if step_type in {"MEMORY", "FALLBACK"}:

                    query = step["content"] or message

                    if step_type == "MEMORY":

                        memories = get_relevant_memories(query)

                        if not memories:

                            print(
                                "JARVIS: I don't have any relevant memories."
                            )

                            continue

                        prompt = build_memory_prompt(
                            query,
                            memories
                        )

                    else:

                        prompt = query

                    print("JARVIS [Local]: ", end="")

                    try:
                        ask_llm(prompt)
                    except Exception as error:
                        print(f"Local model unavailable: {error}")

                elif step_type == "TOOL":

                    try:
                        result = execute_tool(
                            step["action"],
                            step.get("target")
                        )
                    except Exception as error:
                        print(f"Tool execution failed: {error}")
                        continue

                    if result is False:

                        print(
                            "JARVIS: I can't execute that tool."
                        )

                    elif result is not None:

                        print(f"JARVIS: {result}")

                elif step_type == "CLOUD":

                    if mode_state.mode == Mode.LOCAL:
                        # Local mode has no cloud route by design.
                        continue

                    query = step["content"] or message

                    print("JARVIS [Cloud]: ", end="")

                    try:
                        print(ask_gemini(query))
                    except Exception as error:
                        print(f"Cloud service unavailable: {error}")

                elif step_type == "BLOCKED":

                    print(
                        "JARVIS: I can't help with that request."
                    )

    finally:

        # Queue only after response execution/output completes.
        # The daemon writer keeps SQLite work off the response path.
        enqueue_interaction(message)

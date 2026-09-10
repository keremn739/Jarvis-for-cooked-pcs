import sqlite3

def add_memory(content):
    connection = sqlite3.connect("data/jarvis.db")
    connection.execute("INSERT INTO memories (content) VALUES (?)", (content,))

    connection.commit()
    connection.close()


def get_memories():
    connection = sqlite3.connect("data/jarvis.db")

    memories = connection.execute("SELECT content FROM memories").fetchall()
    connection.close()
    return memories

def delete_memory(memory_id):
    connection = sqlite3.connect("data/jarvis.db")
    connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

    connection.commit()
    connection.close()

def find_memories(search):
    connection = sqlite3.connect("data/jarvis.db")
    memory = connection.execute("SELECT content FROM memories WHERE content LIKE ?", (f"%{search}%",)).fetchall()
    connection.close()
    return memory



# ==========================================
# ingest.py - שלב ההכנה (Offline)
# קורא את ספר המתכונים, מחלק אותו למתכונים,
# יוצר Embeddings ושומר הכל ב-ChromaDB.
# מריצים פעם אחת (ושוב רק אם ספר המתכונים השתנה):
#   python ingest.py
# ==========================================

import os

import truststore
truststore.inject_into_ssl()  # משתמש בתעודות של Windows (בגלל הסינון)

import chromadb
from docx import Document as WordDocument
from dotenv import load_dotenv

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


# ==========================================
# הגדרות
# ==========================================

RECIPES_FILE = "data/recipes.docx"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "recipes"
EMBEDDING_MODEL = "gemini-embedding-001"

# רשת ביטחון: מתכון רגיל נכנס כולו ל-chunk אחד.
# רק מתכון ארוך במיוחד יחולק לכמה חלקים.
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100


# ==========================================
# 1. קריאת ספר המתכונים לפי כותרות
# ==========================================

def read_recipes(file_path):
    """
    עוברת על הפסקאות בקובץ ה-Word:
    - כותרת 1 (Heading 1) = קטגוריה, למשל "מאפים פרווה"
    - כותרת 2 (Heading 2) = שם מתכון
    - כל שאר הפסקאות = תוכן המתכון הנוכחי
    מחזירה רשימה של מתכונים.
    """
    word_file = WordDocument(file_path)

    recipes = []
    category = "ללא קטגוריה"
    current = None

    for paragraph in word_file.paragraphs:
        text = paragraph.text.strip()
        style = paragraph.style.name

        if not text:
            continue

        if style == "Heading 1":
            category = text

        elif style == "Heading 2":
            current = {"title": text, "category": category, "lines": []}
            recipes.append(current)

        elif current is not None:
            current["lines"].append(text)

    # מסננים מתכונים בלי תוכן
    return [recipe for recipe in recipes if recipe["lines"]]


# ==========================================
# 2. הפיכת כל מתכון ל-Document של LlamaIndex
# ==========================================

def build_documents(recipes):
    """
    לכל מתכון יוצרים Document עם:
    - text: שם המתכון + התוכן שלו
    - metadata: מידע נלווה לסינון ולציון מקור
    """
    documents = []

    for recipe in recipes:
        text = recipe["title"] + "\n" + "\n".join(recipe["lines"])

        documents.append(
            Document(
                text=text,
                metadata={
                    "source": os.path.basename(RECIPES_FILE),
                    "category": recipe["category"],
                    "recipe_title": recipe["title"],
                },
            )
        )

    return documents


# ==========================================
# 3. הגדרת מודל ה-Embeddings (Gemini - חינמי)
# ==========================================

def setup_embedding():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or "כאן_להדביק" in api_key:
        raise ValueError("חסר GEMINI_API_KEY בקובץ .env")

    Settings.embed_model = GoogleGenAIEmbedding(
        model_name=EMBEDDING_MODEL,
        api_key=api_key,
        embed_batch_size=50,  # 50 מתכונים בכל בקשה - חוסך בקשות במכסה החינמית
    )


# ==========================================
# 4. הכנת ChromaDB
# ==========================================

def setup_vector_store():
    client = chromadb.PersistentClient(path=DB_PATH)

    # מוחקים את האוסף הקודם (אם יש), כדי שהרצה חוזרת
    # לא תשמור כל מתכון פעמיים
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    store = ChromaVectorStore(chroma_collection=collection)
    return StorageContext.from_defaults(vector_store=store), collection


# ==========================================
# 5. שמירה: חלוקה + Embeddings + אינדקס
# ==========================================

def save_documents(documents, storage_context):
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True,
    )


# ==========================================
# הפונקציה הראשית
# ==========================================

def main():
    load_dotenv()  # קורא את המפתח מקובץ .env

    recipes = read_recipes(RECIPES_FILE)
    print(f"נמצאו {len(recipes)} מתכונים")

    documents = build_documents(recipes)

    setup_embedding()
    storage_context, collection = setup_vector_store()

    save_documents(documents, storage_context)

    print(f"נשמרו {collection.count()} קטעים ב-ChromaDB")


if __name__ == "__main__":
    main()
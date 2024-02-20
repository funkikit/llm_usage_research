# pip install pycryptodome
import os
from glob import glob
from pathlib import Path

# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_community.llms import OpenAI
# from langchain.llms import OpenAI

from langchain_community.callbacks import get_openai_callback
# from langchain.callbacks import get_openai_callback

# from PyPDF2 import PdfReader
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Qdrant
# from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


from langchain.docstore.document import Document
# from langchain.document_loaders.base import BaseLoader
from langchain_community.document_loaders.base import BaseLoader
# from langchain_community.document_loaders.base import BaseBlobParser, BaseLoader
# from langchain.document_loaders import UnstructuredHTMLLoader

from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import BSHTMLLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

from tqdm import tqdm
from typing import Dict, List, Union, Tuple
import logging
logger = logging.getLogger()

QDRANT_PATH = "./local_qdrant"
COLLECTION_NAME = "my_collection_2"
# COLLECTION_NAME = "my_documents"

from bs4 import BeautifulSoup

def extract_text(soup: BeautifulSoup) -> Tuple[List[str], List[str]]:
    """
    BeautifulSoupを使って文章とHタグのリストを返す
    """
    structure_sentences = []
    topic_path_list = []
    h1_headings = soup.find_all(['h1'])

    for h1_heading in h1_headings:
        h1_sibling = h1_heading.find_next_sibling()
        h2_headings = h1_sibling.find_all(['h2'])

        for h2_heading in h2_headings:
            h2_sibling = h2_heading.find_next_sibling()
            h3_headings = h2_sibling.find_all(['h3'])

            if len(h3_headings) == 0:
                topic_str = h1_heading.get_text() + " > " + h2_heading.get_text()
                sentence = h2_sibling.get_text()
            else:
                for h3_heading in h3_headings:
                    h3_sibling = h3_heading.find_next_sibling()
                    topic_str = h1_heading.get_text() + " > " + h2_heading.get_text()
                    sentence = ""
                    while h3_sibling:
                        sentence += h3_sibling.text.strip()
                        h3_sibling = h3_sibling.find_next_sibling()

            sentence = sentence.replace("\n", " ")
            sentence = sentence.replace("\u3000", " ")
            structure_sentences.append(sentence)
            topic_path_list.append(topic_str)
        
        return structure_sentences, topic_path_list
    
class StructuredHTMLLoader(BaseLoader):
    """Loader that uses beautiful soup to parse HTML files."""
    def __init__(
        self,
        file_path: str,
        open_encoding: Union[str, None] = None,
        bs_kwargs: Union[dict, None] = None,
    ) -> None:
        """Initialise with path, and optionally, file encoding to use, and any kwargs
        to pass to the BeautifulSoup object."""
        try:
            import bs4  # noqa:F401
        except ImportError:
            raise ValueError(
                "bs4 package not found, please install it with " "`pip install bs4`"
            )

        self.file_path = file_path
        self.open_encoding = open_encoding
        if bs_kwargs is None:
            bs_kwargs = {"features": "lxml"}
        self.bs_kwargs = bs_kwargs

    def load(self) -> List[Document]:
        """Load HTML document into document objects."""
        with open(self.file_path, "r", encoding=self.open_encoding) as f:
            soup = BeautifulSoup(f, **self.bs_kwargs)

        text_list, topic_path_list = extract_text(soup)
            
        if soup.title:
            title = str(soup.title.string)
        else:
            title = ""

        metadata_list = []
        for topic_path in topic_path_list:
            metadata_list.append({
                "source": str(self.file_path),
                "title": title,
                "topic_path": topic_path
            })
        ret = [Document(
            page_content=text, metadata=metadata
        ) for text, metadata in zip(text_list, metadata_list)]
        return ret
    

def add_topic_path_to_text(doc):
    doc.page_content = doc.metadata["topic_path"] + " : " + doc.page_content
    return doc


def get_html_text():
    print(os.getcwd())
    # html_dir = Path("./html/downloaded_html_files/")
    html_dir = Path("./html/add_1/")
    
    if not html_dir.exists():
        raise FileNotFoundError(f"Directory '{html_dir}' does not exist.")
    html_files = list(html_dir.glob("*html"))
    all_documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    for p in tqdm(html_files, total=len(html_files)):
        # StructuredHTMLLoaderでHTMLを読み込み
        doc = StructuredHTMLLoader(p).load()
        # text splitterで一定文字数以内に分割
        print(doc)
        all_documents += text_splitter.split_documents(doc)
    # metadataに格納していたHタグの情報を各テキストに付与
    all_documents = [add_topic_path_to_text(doc) for doc in all_documents]
    # 下記でOpenAIでembedding用に推奨されている"text-embedding-ada-002"が指定されます
    # embeddings = OpenAIEmbeddings()
    return all_documents

    # print(os.getcwd())
    # html_dir = Path("./html/downloaded_html_files/")
    # if not html_dir.exists():
    #     raise FileNotFoundError(f"Directory '{html_dir}' does not exist.")
    # html_files = list(html_dir.glob("*html"))

    # all_documents = []
    # # HTML内のデータを分割する用のSplitterを用意
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)

    # for p in html_files:
    #     # langchainに用意されたHTMLLoaderでHTMLからdocumentを抽出
    #     doc = UnstructuredHTMLLoader(p).load()
    #     # doc = BSHTMLLoader(p).load()
    #     # documentをsplitterで分割
    #     print(doc)
    #     all_documents += text_splitter.split_documents(doc)
    # return all_documents


# def get_pdf_text():
#     uploaded_file = st.file_uploader(
#         label='Upload your PDF here😇',
#         type='pdf'
#     )
#     if uploaded_file:
#         pdf_reader = PdfReader(uploaded_file)
#         text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])
#         text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#             model_name="text-embedding-ada-002",
#             # 適切な chunk size は質問対象のPDFによって変わるため調整が必要
#             # 大きくしすぎると質問回答時に色々な箇所の情報を参照することができない
#             # 逆に小さすぎると一つのchunkに十分なサイズの文脈が入らない
#             chunk_size=500,
#             chunk_overlap=0,
#         )
#         return text_splitter.split_text(text)
#     else:
#         return None


def load_qdrant():
    client = QdrantClient(path=QDRANT_PATH)
    # すべてのコレクション名を取得
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    # コレクションが存在しなければ作成
    if COLLECTION_NAME not in collection_names:
        # コレクションが存在しない場合、新しく作成します
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print('collection created')

    return Qdrant(
        client=client,
        collection_name=COLLECTION_NAME, 
        embeddings=OpenAIEmbeddings()
    )


def build_vector_store(text):
    qdrant = load_qdrant()
    qdrant.add_documents(text)
    
    # テキストをベクトルに変換するモデルをロード
    # embeddings = OpenAIEmbeddings()
    # all_documentsをベクトルに変換
    # vectorized_documents = embeddings.embed_documents(text)
    # # ベクトルをQdrantに格納
    # for i, vector in enumerate(vectorized_documents):
    #     qdrant.upsert_points(collection_name="your_collection_name", 
    #                         points=[UpsertPoint(id=i, vector=vector)])

    # 以下のようにもできる。この場合は毎回ベクトルDBが初期化される
    # LangChain の Document Loader を利用した場合は `from_documents` にする
    # Qdrant.from_documents(
    #     text,
    #     OpenAIEmbeddings(),
    #     path="./local_qdrant",
    #     collection_name="my_documents",
    # )
    
# def page_pdf_upload_and_build_vector_db():
#     st.title("PDF Upload")
#     container = st.container()
#     with container:
#         pdf_text = get_pdf_text()
#         if pdf_text:
#             with st.spinner("Loading PDF ..."):
#                 build_vector_store(pdf_text)

def html_loder_build_vector_db():
    html_text = get_html_text()
    build_vector_store(html_text)

def main():
    html_loder_build_vector_db()

if __name__ == '__main__':
    main()
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

import requests
from bs4 import BeautifulSoup
from langchain_community.embeddings.openai import OpenAIEmbeddings
from chroma import Chroma
# WebScraperの作成
from langchain import WebScraper
from pathlib import Path
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import logging
from typing import Dict, List, Union


from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

# Example for document loading (from url), splitting, and creating vectostore
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
    
logger = logging.getLogger()

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
        from bs4 import BeautifulSoup

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

html_dir = Path("rag_test_sazae/html/downloaded_html_files")
html_files = list(html_dir.glob("*html"))
all_documents = []
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
for p in tqdm(html_files, total=len(html_files)):
    # StructuredHTMLLoaderでHTMLを読み込み
    doc = StructuredHTMLLoader(p).load()
    # text splitterで一定文字数以内に分割
    all_documents += text_splitter.split_documents(doc)

def add_topic_path_to_text(doc):
    doc.page_content = doc.metadata["topic_path"] + " : " + doc.page_content
    return doc

# metadataに格納していたHタグの情報を各テキストに付与
all_documtents = [add_topic_path_to_text(doc) for doc in all_documents]

# 下記でOpenAIでembedding用に推奨されている"text-embedding-ada-002"が指定されます
embeddings = OpenAIEmbeddings()
# Langchainでデフォルトで使われる Chroma という VectorStore を利用
db = Chroma.from_documents(all_documents, embeddings, persist_directory="exp_for_blog")

retriever = db.as_retriever()

# RAG prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
model = ChatOpenAI()

# RAG chain
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)

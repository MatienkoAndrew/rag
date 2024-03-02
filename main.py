from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

import os
from getpass import getpass
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import GigaChat
from langchain.chains import RetrievalQA

import argparse
import logging
import time
from omegaconf import DictConfig, OmegaConf
import hydra

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


##-- global funcs
def log_function_start_end(logger):
    """Декоратор для логирования начала и конца выполнения функции."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"Starting {func.__name__}...")
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            logger.info(f"Finished {func.__name__}. in {int(hours)}h {int(minutes)}min {int(seconds)}s")
            return result
        return wrapper
    return decorator


def parse_args():
    """Функция для парсинга аргументов"""
    parser = argparse.ArgumentParser(description="Question Answering System for PDF documents.")
    parser.add_argument("pdf_document", type=str, nargs='?', default="impromptu-rh.pdf",
                        help="Path to the PDF document for analysis. Defaults to 'impromptu-rh.pdf'.")
    return parser.parse_args()


@log_function_start_end(logger)
def process_pdf(file_path):
    """Функция для чтения и обработки PDF документа"""
    doc_reader = PdfReader(file_path)
    raw_text = ''.join(page.extract_text() or '' for page in doc_reader.pages)
    return raw_text


@log_function_start_end(logger)
def create_docsearch(texts):
    """Функция для создания векторного пространства и документного поиска"""
    embeddings = SentenceTransformerEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    return docsearch


@log_function_start_end(logger)
def perform_qa_queries(queries, docsearch):
    """Функция для выполнения запросов к QA системе"""
    llm = GigaChat(verify_ssl_certs=False)
    for query in queries:
        try:
            retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            rqa = RetrievalQA.from_chain_type(llm=llm,  # OpenAI(),
                                              chain_type="stuff",
                                              retriever=retriever,
                                              return_source_documents=True)
            response = rqa(query)['result']
            logger.info(f"Question: {query}\nAnswer: {response}")
        except Exception as e:
            logger.error(f"Failed to process query '{query}': {e}")


@hydra.main(version_base=None, config_path="config/", config_name="config")
def main(config: DictConfig):
    credentials_model = OmegaConf.to_container(config['model'], resolve=True)
    app_config = OmegaConf.to_container(config['app'], resolve=True)
    os.environ["GIGACHAT_CREDENTIALS"] = credentials_model['GIGACHAT_CREDENTIALS']

    # Парсинг аргументов
    # args = parse_args()
    # pdf_file_path = args.pdf_document

    # Обработка PDF документа
    raw_text = process_pdf(file_path=app_config['filepath'])

    # Создание чанков текста для индексации
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(raw_text)

    # Создание векторного пространства
    docsearch = create_docsearch(texts)

    # Примеры запросов
    queries = [
        "how does GPT-4 change social media?",
        "who are the authors of the book?",
        "что такое OpenAI?"
    ]

    ##-- Retrieval Question Answering system
    perform_qa_queries(queries, docsearch)


if __name__ == '__main__':
    main()

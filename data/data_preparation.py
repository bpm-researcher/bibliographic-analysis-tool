import streamlit as st
import tempfile
from io import BytesIO
import os


CROSSREF_AVAILABLE_FIELDS = {
    'author': 'author',
    'Title': 'title',
    'Abstract': 'abstract',
    'Language': 'language',
    'Article References': 'reference',
    'Author References': 'reference',
    'Times Cited': 'is-referenced-by-count',
    'Publication Year': 'created',
    'DOI': 'DOI',
    'Publisher': 'publisher'
}

REFERENCE_FIELDS = ["DOI", "article-title", "author"]

def show():
    st.set_page_config(page_title="Cross-ref helper", page_icon="🔎")
    st.title("Cross-ref Excel helper")

    operation_label = st.radio(
        "Choose an operation",
        list(OPERATIONS.keys())
    )

    uploaded_file = st.file_uploader(
        "Upload the Excel file you want to process (.xlsx)",
        type=["xlsx"]
    )

    citation_field = None
    if operation_label == "Fill ONE missing column":
        citation_field = st.selectbox(
            "Column to complete",
            list(CROSSREF_AVAILABLE_FIELDS.keys())
        )

    run = st.button("Run")

    if run:

        if uploaded_file is None:
            st.warning("Please upload an Excel file first.")
            st.stop()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_in:
            tmp_in.write(uploaded_file.getbuffer())
            input_path = tmp_in.name

        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        output_path = tmp_out.name
        tmp_out.close()            

        try:
            if operation_label == "Fill ONE missing column":
                fill_missing_field(input_path, citation_field, output_path)
            else:
                fill_missing_fields(input_path, output_path)

            with open(output_path, "rb") as f:
                result_bytes = f.read()

            st.success("Processing finished! Download the file below.")
            st.download_button(
                label="Download processed Excel file",
                data=result_bytes,
                file_name="processed.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"Error during processing: {e}")

        finally:
            for path in (input_path, output_path):
                try:
                    os.remove(path)
                except OSError:
                    pass

def get_field_from_api(crossref_field, search_term):
    url = "https://api.crossref.org/works/" + search_term
    try:
        response = requests.get(url,timeout=10)
        response.raise_for_status()
        data = response.json()
        items = data['message']
        if items:
            
            return items[crossref_field]
    except requests.RequestException as e:
        print(f"Request failed for search of: {search_term}\nError: {e}")
        return None        


def parse_article_references(article, references):
    references_line = ""
    for reference in references:
        reference_text = ""
        if reference_fields[1] in reference:
            reference_text = reference[reference_fields[1]] + "; "
        elif reference_fields[0] in reference:
            try:
                title = get_field_from_api('title', reference[reference_fields[0]])
                if title:
                    reference_text = title[0] + "; "
            except Exception as e:
                print(f"Could Not Find Title For article: {e}")
                reference_text = reference[reference_fields[0]] + "; "
        elif "unstructured" in reference:
            reference_text = reference['unstructured'] + "; "
        else:
            error_articles(article, reference)
            continue
        references_line = references_line + reference_text 
    return references_line

def parse_year(date):
    return date['date-parts'][0][0]

def parse_field_value(crossref_field, field_value, citation_field, article):
    if crossref_field == "title":
        field_value = field_value[0]
    if crossref_field == "reference":
        if citation_field == 'Article References':
            field_value = parse_article_references(article, field_value)
    if crossref_field == "created":
        field_value = parse_year(field_value)
    return field_value

def fill_missing_field(excel_path, citation_field, output_path):
    
    df = pd.read_excel(excel_path)
    try:
        df = process_each_field(citation_field, df)
    except Exception as e:
        print(f"Unexpected error while processing field {citation_field}: {e}")
    df.to_excel(output_path, index=False)
    print(f"Output saved to {output_path}")

def fill_missing_fields(excel_path, output_path):
    df = pd.read_excel(excel_path)
    for key in crossref_fields:
        try:
            df = process_each_field(key, df)
        except Exception as e:
            print(f"Unexpected error while processing fields: {e}")
        df.to_excel(output_path, index=False)
        print(f"Output saved to {output_path}")

def process_each_field(citation_field, df):
    
    crossref_field = crossref_fields[citation_field]
    for index, citation in df.iterrows():
        if pd.isna(citation[citation_field]) or str(citation[citation_field]).strip() == '':
            if crossref_field == "DOI":
                search_term = citation["Title"]
            else:
                search_term = citation['DOI']
            try:
                
                field_value = get_field_from_api(crossref_field, search_term)
                if field_value:
                    field_value = parse_field_value(crossref_field, field_value, citation_field, citation["Title"])
                    df.at[index, citation_field] = field_value
                    print(f" → Found %s: {field_value}", citation_field)
                    field_value = None
            except Exception as e:
                print(f"Unable to get data from API: {e}")
            finally:
                continue
                
    return df

def error_articles(article_name, reference):
    with open(
        "biliographic_analysis_on_indicators/data_preparation/outputs/error.txt",
        "a",
        encoding="utf-8"
    ) as f:
        f.write(article_name + "\n")
        if isinstance(reference, dict):
            f.write(json.dumps(reference, ensure_ascii=False) + "\n")
        else:
            f.write(str(reference) + "\n")

OPERATIONS = {
    "Fill ONE missing column"  : fill_missing_field,
    "Fill ALL missing columns" : fill_missing_fields
}
    











from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode


class MedicalParser:
    def __init__(self):
        self.pipeline_options = PdfPipelineOptions(do_table_structure=True)
        self.pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    def convert_to_markdown(self, pdf_path, start_page=None, end_page=None):
        """
        Converts a PDF to Markdown.

        Args:
            pdf_path (str): Path to the PDF file.
            start_page (int, optional): 0-based start page index (will be converted to 1-based for Docling).
            end_page (int, optional): 0-based end page index (will be converted to 1-based for Docling).

        Returns:
            str: Markdown content of the document.
        """
        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )

        # Docling uses 1-based page numbers; our config uses 0-based indices
        page_range = None
        if start_page is not None or end_page is not None:
            page_start = (start_page + 1) if start_page is not None else 1
            page_end = (end_page + 1) if end_page is not None else None
            # page_range as tuple (start, end); pass None end means "until last page"
            if page_end is not None:
                page_range = (page_start, page_end)
            else:
                page_range = (page_start,)

        # Some Docling versions accept 'pages' as a list; others accept 'page_range' as a tuple.
        # We try the tuple approach first; if your version differs, switch to:
        #   result = converter.convert(pdf_path, pages=list(range(page_start, page_end + 1)))
        try:
            if page_range and len(page_range) == 2:
                result = converter.convert(pdf_path, page_range=page_range)
            elif page_range and len(page_range) == 1:
                # Only start page defined — convert from start_page to end of document
                result = converter.convert(pdf_path, page_range=(page_range[0], 9999))
            else:
                result = converter.convert(pdf_path)
        except TypeError:
            # Fallback: some Docling versions don't support page_range kwarg
            print(f"⚠️  page_range not supported by this Docling version. Converting full document for: {pdf_path}")
            result = converter.convert(pdf_path)

        md_content = result.document.export_to_markdown()
        return md_content
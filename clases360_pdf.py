"""Conversión segura de hojas OMR escaneadas para Clases360."""
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
from flask import request, jsonify, send_file

def activar_pdf_clases360(app):
    @app.route('/clases360/pdf_a_png', methods=['POST'])
    def pdf_a_png_clases360():
        import fitz
        uploaded = request.files.get('archivo')
        if uploaded is None:
            return jsonify(ok=False, error='Falta archivo PDF'), 400
        data = uploaded.read(25 * 1024 * 1024 + 1)
        if not data.startswith(b'%PDF-') or len(data) > 25 * 1024 * 1024:
            return jsonify(ok=False, error='PDF inválido o demasiado grande'), 400
        try:
            doc = fitz.open(stream=data, filetype='pdf')
            if doc.is_encrypted or not 1 <= len(doc) <= 60:
                return jsonify(ok=False, error='PDF cifrado o más de 60 páginas'), 422
            out = BytesIO()
            with ZipFile(out, 'w', ZIP_DEFLATED, compresslevel=2) as archive:
                for index, page in enumerate(doc):
                    pixels = page.get_pixmap(matrix=fitz.Matrix(200/72,200/72), alpha=False)
                    if pixels.width*pixels.height > 18000000:
                        return jsonify(ok=False, error='Página demasiado grande'), 422
                    archive.writestr(f'pagina_{index+1:03d}.png', pixels.tobytes('png'))
                    if out.tell() > 80 * 1024 * 1024:
                        return jsonify(ok=False, error='PDF convertido demasiado grande'), 422
            out.seek(0)
            return send_file(out, mimetype='application/zip', as_attachment=True, download_name='paginas_omr.zip')
        except Exception:
            app.logger.exception('Error conversión PDF Clases360')
            return jsonify(ok=False, error='No se pudo convertir el PDF'), 422

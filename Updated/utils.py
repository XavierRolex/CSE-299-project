from fpdf import FPDF

def export_chat_to_pdf(chat_history):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for entry in chat_history:
        role = "You" if entry["role"] == "user" else "Bot"
        pdf.multi_cell(0, 10, f"{role}: {entry['text']}\n", align='L')

    return pdf.output(dest='S').encode('latin-1')

import os
import uuid
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)
from config import REPORTS_FOLDER, UPLOAD_FOLDER, MASKS_FOLDER, HEATMAPS_FOLDER

def generate_pdf_report(prediction_data, kb_data=None):
    """
    Generates a professional, print-ready PDF diagnostic report for a cocoa scan.
    prediction_data contains:
      pred_uuid, timestamp, image_filename, mask_filename, heatmap_filename,
      disease_name, predicted_key, confidence, severity, affected_area_pct,
      is_uncertain, environmental_risk
    """
    report_uuid = prediction_data.get('pred_uuid', f"rep_{uuid.uuid4().hex[:8]}")
    pdf_filename = f"report_{report_uuid}.pdf"
    pdf_path = os.path.join(REPORTS_FOLDER, pdf_filename)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom Palette
    c_primary = colors.HexColor("#3b2a1a")      # Dark cocoa brown
    c_gold = colors.HexColor("#c48c40")         # Cocoa gold
    c_gold_light = colors.HexColor("#fdf7eb")   # Warm parchment
    c_dark = colors.HexColor("#222222")
    c_muted = colors.HexColor("#666666")
    c_danger = colors.HexColor("#b02a37")
    c_warning = colors.HexColor("#997404")
    c_success = colors.HexColor("#146c43")

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=c_primary
    )

    sub_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=c_muted
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=c_primary,
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=c_dark
    )

    badge_style = ParagraphStyle(
        'Badge_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.white
    )

    alert_style = ParagraphStyle(
        'Alert_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=c_danger
    )

    story = []

    # 1. Header Banner Table
    header_data = [
        [
            Paragraph("<b>COCOA GUARD</b> | AI Plant Health Monitoring System", title_style),
            Paragraph(f"<b>REPORT ID:</b> #{report_uuid.upper()}<br/><b>DATE:</b> {prediction_data.get('timestamp', datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))}", sub_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[360, 180])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=2, color=c_gold, spaceBefore=4, spaceAfter=8))

    # 2. Farmer Emergency Alert (if Severe or High Risk)
    severity = prediction_data.get('severity', 'Healthy')
    env_risk = prediction_data.get('environmental_risk', 'Low')
    is_severe = (severity == 'Severe' or env_risk == 'High')

    if is_severe:
        alert_text = (
            "<b>CRITICAL PHYTOSANITARY WARNING: Immediate attention recommended.</b><br/>"
            "High pathogen severity (>35% tissue necrosis) or severe environmental risk detected. "
            "Implement containment and quarantine protocols to prevent epidemic transmission."
        )
        alert_data = [[Paragraph(alert_text, alert_style)]]
        alert_table = Table(alert_data, colWidths=[540])
        alert_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8d7da")),
            ('TEXTCOLOR', (0, 0), (-1, -1), c_danger),
            ('BOX', (0, 0), (-1, -1), 1, c_danger),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ]))
        story.append(alert_table)
        story.append(Spacer(1, 8))

    # 3. Diagnostic & Severity Summary Cards
    disease_name = prediction_data.get('disease_name', 'Unknown')
    conf = prediction_data.get('confidence', 0.0)
    affected_pct = prediction_data.get('affected_area_pct', 0.0)

    summary_rows = [
        [
            Paragraph("<b>Diagnosis</b>", sub_style),
            Paragraph("<b>Confidence Score</b>", sub_style),
            Paragraph("<b>Severity Classification</b>", sub_style),
            Paragraph("<b>Affected Area (%)</b>", sub_style)
        ],
        [
            Paragraph(f"<b>{disease_name}</b>", h2_style),
            Paragraph(f"<b>{conf}%</b>", h2_style),
            Paragraph(f"<b>{severity}</b>", h2_style),
            Paragraph(f"<b>{affected_pct}%</b>", h2_style)
        ]
    ]
    summary_table = Table(summary_rows, colWidths=[160, 120, 140, 120])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_gold_light),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#e0d4be")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e0d4be")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 10))

    # 4. Visual Triad: Original Image, Segmentation Mask, Grad-CAM Heatmap
    story.append(Paragraph("<b>Computer Vision & Explainable AI Triad</b>", h2_style))

    orig_img_path = os.path.join(UPLOAD_FOLDER, prediction_data.get('image_filename', ''))
    mask_name = prediction_data.get('mask_filename', '')
    overlay_path = os.path.join(MASKS_FOLDER, mask_name.replace('mask_', 'overlay_').replace('.png', '.jpg'))
    if not os.path.exists(overlay_path):
        overlay_path = os.path.join(MASKS_FOLDER, mask_name)

    heatmap_path = os.path.join(HEATMAPS_FOLDER, prediction_data.get('heatmap_filename', ''))

    triad_cells = []
    labels = []

    # Original
    if os.path.exists(orig_img_path):
        triad_cells.append(RLImage(orig_img_path, width=170, height=130))
    else:
        triad_cells.append(Paragraph("Original Image Not Available", sub_style))
    labels.append(Paragraph("<para align=center><b>1. Input Field Specimen</b></para>", sub_style))

    # Segmentation
    if os.path.exists(overlay_path):
        triad_cells.append(RLImage(overlay_path, width=170, height=130))
    elif os.path.exists(os.path.join(MASKS_FOLDER, mask_name)):
        triad_cells.append(RLImage(os.path.join(MASKS_FOLDER, mask_name), width=170, height=130))
    else:
        triad_cells.append(Paragraph("Segmentation Mask Not Available", sub_style))
    labels.append(Paragraph("<para align=center><b>2. Lesion Segmentation Mask</b></para>", sub_style))

    # GradCAM
    if os.path.exists(heatmap_path):
        triad_cells.append(RLImage(heatmap_path, width=170, height=130))
    else:
        triad_cells.append(Paragraph("Grad-CAM Heatmap Not Available", sub_style))
    labels.append(Paragraph("<para align=center><b>3. Grad-CAM Attention Heatmap</b></para>", sub_style))

    triad_table = Table([triad_cells, labels], colWidths=[180, 180, 180])
    triad_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(triad_table)
    story.append(Spacer(1, 10))

    # 5. Treatment Recommendations & Management Protocol
    if kb_data:
        story.append(Paragraph("<b>Prescriptive Agricultural Management Protocol</b>", h2_style))

        # Symptoms
        symptoms_list = kb_data.get('symptoms', [])
        symptom_bullets = "<br/>".join([f"• {s}" for s in symptoms_list[:3]])

        # Chemical
        chem_list = kb_data.get('chemical_fertilizers', [])
        chem_text = "<br/>".join([f"• <b>{c['name']}:</b> {c['description']}" for c in chem_list[:2]])

        # Organic
        org_list = kb_data.get('organic_treatments', [])
        org_text = "<br/>".join([f"• {o}" for o in org_list[:2]])

        # Expert Triggers
        expert_list = kb_data.get('expert_consultation_triggers', [])
        expert_text = "<br/>".join([f"• {e}" for e in expert_list[:2]])

        protocol_rows = [
            [Paragraph("<b>Diagnostic Symptoms:</b>", body_style), Paragraph(symptom_bullets or "No critical symptoms recorded.", body_style)],
            [Paragraph("<b>Therapeutic Spray / Fertilizer:</b>", body_style), Paragraph(chem_text or "Follow standard maintenance schedule.", body_style)],
            [Paragraph("<b>Organic & Biological Control:</b>", body_style), Paragraph(org_text or "Standard biological monitoring.", body_style)],
            [Paragraph("<b>When to Seek Expert Assistance:</b>", body_style), Paragraph(f"<font color='#b02a37'><b>Mandatory Escalation:</b></font><br/>{expert_text}", body_style)]
        ]

        protocol_table = Table(protocol_rows, colWidths=[170, 370])
        protocol_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#f5efe4")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#dcd0be")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#dcd0be")),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(protocol_table)
        story.append(Spacer(1, 8))

    # 6. Disclaimer & Digital Seal Footer
    disclaimer_text = (
        "<b>Phytosanitary & AI Disclaimer:</b> This report is generated automatically by Cocoa Guard deep learning "
        "inference algorithms (MobileNetV2 CNN Architecture). It serves as an assistive field screening and decision-support "
        "tool. Final diagnosis for quarantine regulatory actions should be confirmed by certified plant pathologists."
    )
    story.append(Paragraph(disclaimer_text, sub_style))

    # Build the document
    doc.build(story)
    return pdf_path, pdf_filename

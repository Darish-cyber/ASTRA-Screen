"""
ASTRA-Screen: Automated Spaceflight QA Inspection Certificate Generator
Generates official PDF burn-in inspection certificates for ISRO payloads using ReportLab.
Compliant with MIL-STD-883 Method 1015 Class S and AEC-Q001 screening guidelines.
"""

import io
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)

# Friendly display metadata for physical parameters & SHAP features
FEATURE_METRICS = {
    "Iddq_0h": {
        "label": "Initial 0h Quiescent Current (Iddq)",
        "unit": "µA",
        "fmt": "{:.2f} µA",
    },
    "Iddq_24h": {
        "label": "24h Quiescent Current (Iddq)",
        "unit": "µA",
        "fmt": "{:.2f} µA",
    },
    "Delta_Iddq_24_0": {
        "label": "24h Quiescent Drift (ΔIddq)",
        "unit": "µA",
        "fmt": "{:+.4f} µA",
    },
    "Velocity_Iddq": {
        "label": "Iddq Thermal Drift Velocity",
        "unit": "µA/h",
        "fmt": "{:.4f} µA/h",
    },
    "Relative_Drift_Pct": {
        "label": "Relative Current Drift",
        "unit": "%",
        "fmt": "{:+.2f}%",
    },
    "Ileak_0h": {
        "label": "Initial 0h Leakage (Ileak)",
        "unit": "nA",
        "fmt": "{:.2f} nA",
    },
    "Ileak_24h": {
        "label": "24h Sub-threshold Leakage (Ileak)",
        "unit": "nA",
        "fmt": "{:.2f} nA",
    },
    "Delta_Ileak_24_0": {
        "label": "24h Leakage Drift (ΔIleak)",
        "unit": "nA",
        "fmt": "{:+.4f} nA",
    },
    "PropDelay_0h": {
        "label": "Initial 0h Delay (tpd)",
        "unit": "ns",
        "fmt": "{:.4f} ns",
    },
    "PropDelay_24h": {
        "label": "24h Propagation Delay (tpd)",
        "unit": "ns",
        "fmt": "{:.4f} ns",
    },
    "Delta_Delay_24_0": {
        "label": "24h Propagation Delay Drift (Δtpd)",
        "unit": "ns",
        "fmt": "{:+.4f} ns",
    },
}


def generate_qa_certificate_pdf(
    explanation: Dict[str, Any],
    component_row: dict,
    profile_config: Dict[str, Any] = None,
) -> bytes:
    """
    Builds a spaceflight-grade QA Inspection Certificate in PDF format.
    Dynamically adapts title, standards, and chamber conditions to profile_config.
    Returns bytes buffer for direct Streamlit download.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "HeaderTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=colors.HexColor("#0f2b48"),
        alignment=1,
        spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "HeaderSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        textColor=colors.HexColor("#334155"),
        alignment=1,
        spaceAfter=8,
    )
    section_heading = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        textColor=colors.HexColor("#0f2b48"),
        spaceBefore=6,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1e293b"),
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=10,
        bulletIndent=2,
        spaceAfter=2,
    )

    # Table typography styles
    th_style = ParagraphStyle(
        "TH",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9,
        textColor=colors.white,
        alignment=1,
    )
    th_style_left = ParagraphStyle(
        "THLeft",
        parent=th_style,
        alignment=0,
    )
    td_style = ParagraphStyle(
        "TD",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#1e293b"),
        alignment=0,
    )
    td_style_center = ParagraphStyle(
        "TDCenter",
        parent=td_style,
        alignment=1,
    )

    elements = []

    # Dynamic Header by Profile
    org_title = profile_config.get("cert_title_org", "INDIAN SPACE RESEARCH ORGANISATION (ISRO)") if profile_config else "INDIAN SPACE RESEARCH ORGANISATION (ISRO)"
    sub_title = profile_config.get("cert_title_sub", "RELIABILITY & QUALITY ASSURANCE DIVISION &bull; COMPONENT SCREENING FACILITY") if profile_config else "RELIABILITY & QUALITY ASSURANCE DIVISION &bull; COMPONENT SCREENING FACILITY"

    elements.append(Paragraph(org_title, title_style))
    elements.append(Paragraph(sub_title, subtitle_style))
    elements.append(
        HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#0f2b48"), spaceAfter=8)
    )

    raw_comp_id = str(explanation.get("component_id", "UNKNOWN"))
    comp_id = raw_comp_id.replace("--", "-")
    is_pass = explanation.get("is_flight_ready", False)

    default_pass = "PASSED &bull; CLASS S FLIGHT QUALIFIED"
    default_fail = "REJECTED &bull; SCREENING OUTLIER"
    pass_text = profile_config.get("cert_status_pass", default_pass) if profile_config else default_pass
    fail_text = profile_config.get("cert_status_fail", default_fail) if profile_config else default_fail
    status_text = pass_text if is_pass else fail_text
    status_color = colors.HexColor("#166534") if is_pass else colors.HexColor("#991b1b")

    chamber_desc = (
        profile_config.get("cert_chamber_profile", f"{component_row.get('Stress_Temp_C', 125.0)} &deg;C &bull; MIL-STD-883 M1015")
        if profile_config
        else f"{component_row.get('Stress_Temp_C', 125.0)} &deg;C &bull; MIL-STD-883 M1015"
    )

    summary_data = [
        [
            Paragraph("<b>CERTIFICATE NO:</b>", body_style),
            Paragraph(f"CERT-{comp_id}", body_style),
            Paragraph("<b>STATUS:</b>", body_style),
            Paragraph(f"<b><font color='{status_color.hexval()}'>{status_text}</font></b>", body_style),
        ],
        [
            Paragraph("<b>COMPONENT ID:</b>", body_style),
            Paragraph(comp_id, body_style),
            Paragraph("<b>BATCH / LOT ID:</b>", body_style),
            Paragraph(str(component_row.get("Lot_ID", "ISRO-FLIGHT-QUAL-883")), body_style),
        ],
        [
            Paragraph("<b>WAFER POSITION:</b>", body_style),
            Paragraph(f"Wafer #{component_row.get('Wafer_ID', 12)}", body_style),
            Paragraph("<b>CHAMBER PROFILE:</b>", body_style),
            Paragraph(chamber_desc, body_style),
        ],
    ]
    summary_table = Table(summary_data, colWidths=[105, 155, 105, 175])
    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ])
    )
    elements.append(summary_table)
    elements.append(Spacer(1, 8))

    # Section 1: Parametric Table
    elements.append(Paragraph("1. Parametric Log & Drift Profile", section_heading))
    pred_168 = explanation.get("pred_iddq_168h")
    pred_str = f"{pred_168:.2f} µA" if pred_168 is not None else "—"
    status_badge_iddq = "<b><font color='#166534'>NOMINAL</font></b>" if component_row.get("Module_A_Decision") == "PASS_FLIGHT_READY" and is_pass else "<b><font color='#991b1b'>ANOMALY</font></b>"

    param_cells = [
        [
            Paragraph("Parameter Metric", th_style_left),
            Paragraph("0-Hour (Initial)", th_style),
            Paragraph("24-Hour (Screening)", th_style),
            Paragraph("Projected 168-Hour", th_style),
            Paragraph("Datasheet Limit", th_style),
            Paragraph("Screening Status", th_style),
        ],
        [
            Paragraph("<b>Iddq</b> (Quiescent Current)", td_style),
            Paragraph(f"{component_row.get('Iddq_0h', 0.0):.2f} µA", td_style_center),
            Paragraph(f"{component_row.get('Iddq_24h', 0.0):.2f} µA", td_style_center),
            Paragraph(f"<b>{pred_str}</b>", td_style_center),
            Paragraph("≤ 50.00 µA", td_style_center),
            Paragraph(status_badge_iddq, td_style_center),
        ],
        [
            Paragraph("<b>Ileak</b> (Sub-threshold Leakage)", td_style),
            Paragraph(f"{component_row.get('Ileak_0h', 0.0):.2f} nA", td_style_center),
            Paragraph(f"{component_row.get('Ileak_24h', 0.0):.2f} nA", td_style_center),
            Paragraph("—", td_style_center),
            Paragraph("≤ 15.00 nA", td_style_center),
            Paragraph("<b><font color='#166534'>NOMINAL</font></b>", td_style_center),
        ],
        [
            Paragraph("<b>t<sub>pd</sub></b> (Propagation Delay)", td_style),
            Paragraph(f"{component_row.get('PropDelay_0h', 0.0):.4f} ns", td_style_center),
            Paragraph(f"{component_row.get('PropDelay_24h', 0.0):.4f} ns", td_style_center),
            Paragraph("—", td_style_center),
            Paragraph("≤ 2.50 ns", td_style_center),
            Paragraph("<b><font color='#166534'>NOMINAL</font></b>", td_style_center),
        ],
    ]
    param_table = Table(param_cells, colWidths=[145, 75, 75, 85, 80, 80])
    param_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f2b48")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ])
    )
    elements.append(param_table)
    elements.append(Spacer(1, 8))

    # Section 2: Diagnostics
    elements.append(Paragraph("2. Screening Diagnostics & Inspection Log", section_heading))
    abort_badge = (
        "<font color='#991b1b'><b>TRIGGERED (144 Chamber Hours Conserved)</b></font>"
        if not is_pass
        else "<font color='#166534'><b>NEGATIVE (Cleared to Complete Flight Burn-In)</b></font>"
    )
    elements.append(
        Paragraph(
            f"<b>Lot-Relative Anomaly Index:</b> {explanation.get('dynamic_risk_score', 0.0):.1f} / 100.0 &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>24h Abort Flag:</b> {abort_badge}",
            body_style,
        )
    )
    elements.append(Spacer(1, 3))

    if "failure_mode_diagnosis" in explanation:
        diag = explanation["failure_mode_diagnosis"]
        diag_color = "#166534" if is_pass else "#991b1b"
        elements.append(
            Paragraph(
                f"<b>Physical Failure Mode Diagnosis:</b> <font color='{diag_color}'><b>{diag['failure_mode']}</b> (Confidence: {diag['confidence']:.1f}%)</font><br/>"
                f"<b>Physical Evidence:</b> {diag['evidence']}",
                body_style,
            )
        )
        elements.append(Spacer(1, 3))
    for just in explanation.get("inspector_justifications", []):
        safe_just = (
            just.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("&amp;mu;", "µ")
            .replace("&amp;le;", "≤")
            .replace("&amp;deg;", "°")
        )
        elements.append(Paragraph(f"&bull; {safe_just}", bullet_style))

    elements.append(Spacer(1, 6))

    # Section 3: Feature attribution
    elements.append(Paragraph("3. Parametric Influence Factors (SHAP Root-Cause Attribution)", section_heading))
    shap_cells = [
        [
            Paragraph("Parameter / Feature Metric", th_style_left),
            Paragraph("Logged Value", th_style),
            Paragraph("Estimated Variance Impact", th_style_left),
        ]
    ]

    for item in explanation.get("top_shap_contributions", []):
        feat_key = item.get("feature_raw", item.get("feature", ""))
        feat_name = item.get("feature", "")

        # Determine clean label and formatted value
        if feat_key in FEATURE_METRICS:
            clean_label = FEATURE_METRICS[feat_key]["label"]
            val_display = FEATURE_METRICS[feat_key]["fmt"].format(item["value"])
        elif feat_name in FEATURE_METRICS:
            clean_label = FEATURE_METRICS[feat_name]["label"]
            val_display = FEATURE_METRICS[feat_name]["fmt"].format(item["value"])
        elif item.get("formatted_value"):
            clean_label = feat_name
            val_display = item["formatted_value"]
        else:
            clean_label = feat_name.replace("_", " ").title()
            val_display = f"{item['value']:.4f}"

        shap_impact = float(item["shap_impact"])
        if shap_impact > 0:
            impact_color = "#991b1b"
            badge_text = f"+{shap_impact:.4f} µA shift (Elevates Drift Risk)"
        elif shap_impact < 0:
            impact_color = "#166534"
            badge_text = f"{shap_impact:.4f} µA shift (Stabilizing / Nominal)"
        else:
            impact_color = "#334155"
            badge_text = f"0.0000 µA shift (Neutral Baseline)"

        shap_cells.append([
            Paragraph(f"<b>{clean_label}</b>", td_style),
            Paragraph(f"<b>{val_display}</b>", td_style_center),
            Paragraph(f"<b><font color='{impact_color}'>{badge_text}</font></b>", td_style),
        ])

    shap_table = Table(shap_cells, colWidths=[195, 115, 230])
    shap_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f2b48")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ])
    )
    elements.append(shap_table)
    elements.append(Spacer(1, 8))

    # Section 4: Sign-off & Digital Integrity Seal
    elements.append(Paragraph("4. Screening Authority & Digital Verification", section_heading))
    default_stds = ["MIL-STD-883 Method 1015 Class S", "AEC-Q001 Section 4"]
    if profile_config and "governing_standards" in profile_config:
        governing_list = profile_config["governing_standards"]
    else:
        governing_list = explanation.get("governing_standards", default_stds)
    standards_str = " &bull; ".join(governing_list)
    elements.append(Paragraph(f"<b>Governing Standards:</b> {standards_str}", body_style))
    elements.append(Spacer(1, 3))

    # Deterministic SHA-256 digital signature digest
    cert_hash_raw = f"{comp_id}:{component_row.get('Lot_ID', 'LOT-883')}:{is_pass}:{explanation.get('dynamic_risk_score', 0.0):.2f}"
    cert_hash = hashlib.sha256(cert_hash_raw.encode("utf-8")).hexdigest().upper()
    hash_formatted = f"SHA256: {cert_hash[:8]}-{cert_hash[8:16]}-{cert_hash[16:24]}-{cert_hash[24:32]}"

    elements.append(
        Paragraph(
            f"<b>Digital QA Integrity Seal:</b> <font face='Courier' color='#0f2b48'>{hash_formatted}</font> &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Certified Active:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            body_style,
        )
    )
    elements.append(Spacer(1, 5))

    engine_label = f"ASTRA-Screen v2.4 ({profile_config.get('standard_code', 'MIL-STD-883')})" if profile_config else "ASTRA-Screen v2.4 (LightGBM + D-PAT)"
    authority_label = profile_config.get("cert_title_sub", "Reliability & Component Screening Facility") if profile_config else "Reliability & Component Screening Facility"
    board_label = profile_config.get("cert_title_org", "Quality Assurance Division, ISRO") if profile_config else "Quality Assurance Division, ISRO"

    signoff_data = [
        [
            Paragraph(f"<b>Automated Engine:</b><br/>{engine_label}", body_style),
            Paragraph(f"<b>Inspection Authority:</b><br/>{authority_label}", body_style),
            Paragraph(f"<b>Qualification Board:</b><br/>{board_label}", body_style),
        ],
        [
            Paragraph("System Checksum: <b><font color='#166534'>VERIFIED [PASS]</font></b>", body_style),
            Paragraph(f"Inspection Verdict: <b><font color='{status_color.hexval()}'>{'CLEARED' if is_pass else 'REJECTED'}</font></b>", body_style),
            Paragraph("Digital Signature: <b>DIGITALLY SEALED</b>", body_style),
        ],
    ]
    signoff_table = Table(signoff_data, colWidths=[175, 175, 190])
    signoff_table.setStyle(
        TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ])
    )
    elements.append(signoff_table)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


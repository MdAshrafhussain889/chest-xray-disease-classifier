
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import io

class MedicalReportGenerator:
    """Generate professional medical PDF reports"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e3a8a'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2563eb'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=11,
            leading=14,
            spaceAfter=10
        ))
    
    def generate_report(self, results, image_path, patient_info, output_path):
        """Generate comprehensive medical report"""
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=letter,
                                rightMargin=0.75*inch, leftMargin=0.75*inch,
                                topMargin=0.75*inch, bottomMargin=0.75*inch)
        
        story = []
        
        # Header
        story.append(Paragraph("CHEST X-RAY ANALYSIS REPORT", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))
        
        # Hospital/Clinic Info (Optional)
        header_data = [
            ['Report ID:', patient_info.get('report_id', 'N/A')],
            ['Date:', datetime.now().strftime('%B %d, %Y at %I:%M %p')],
            ['Patient ID:', patient_info.get('patient_id', 'N/A')],
            ['Patient Name:', patient_info.get('patient_name', 'Anonymous')],
            ['Age/Sex:', f"{patient_info.get('age', 'N/A')} / {patient_info.get('sex', 'N/A')}"]
        ]
        
        header_table = Table(header_data, colWidths=[1.5*inch, 4.5*inch])
        header_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        
        story.append(header_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Analysis Results Section
        story.append(Paragraph("AI-ASSISTED DIAGNOSTIC FINDINGS", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.1*inch))
        
        if len(results) == 0:
            story.append(Paragraph(
                "✓ <b>No abnormalities detected.</b> The AI analysis did not identify any significant pathological findings.",
                self.styles['BodyText']
            ))
        else:
            story.append(Paragraph(
                f"⚠ <b>Found {len(results)} potential abnormality(ies):</b>",
                self.styles['BodyText']
            ))
            story.append(Spacer(1, 0.1*inch))
            
            # Results table
            table_data = [['Disease', 'Confidence', 'Threshold', 'Status']]
            
            for res in results:
                confidence_pct = f"{res['confidence']*100:.1f}%"
                threshold_pct = f"{res['threshold']*100:.1f}%"
                status = "DETECTED" if res['detected'] else "Not Detected"
                
                table_data.append([
                    res['disease'],
                    confidence_pct,
                    threshold_pct,
                    status
                ])
            
            results_table = Table(table_data, colWidths=[2.5*inch, 1.3*inch, 1.3*inch, 1.5*inch])
            results_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            
            story.append(results_table)
        
        story.append(Spacer(1, 0.3*inch))
        
        # X-Ray Image
        story.append(Paragraph("CHEST X-RAY IMAGE", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.1*inch))
        
        img = Image(image_path, width=4*inch, height=4*inch)
        story.append(img)
        story.append(Spacer(1, 0.3*inch))
        
        # Clinical Notes
        story.append(Paragraph("CLINICAL INTERPRETATION", self.styles['SectionHeader']))
        story.append(Paragraph(
            "This report is generated using a DenseNet121-based deep learning model trained on the "
            "NIH Chest X-ray14 dataset. The model provides probability scores for 14 different thoracic diseases. "
            "<b>This is an AI-assisted screening tool and should be reviewed by a qualified radiologist "
            "before making any clinical decisions.</b>",
            self.styles['BodyText']
        ))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Recommendations
        if len(results) > 0:
            story.append(Paragraph("RECOMMENDATIONS", self.styles['SectionHeader']))
            story.append(Paragraph(
                "• Follow-up with a board-certified radiologist for comprehensive evaluation<br/>"
                "• Clinical correlation with patient history and symptoms recommended<br/>"
                "• Consider additional imaging studies if clinically indicated<br/>"
                "• Correlate findings with physical examination and laboratory results",
                self.styles['BodyText']
            ))
        
        story.append(Spacer(1, 0.4*inch))
        
        # Footer
        story.append(Paragraph("_" * 80, self.styles['BodyText']))
        story.append(Paragraph(
            "<i>This report is computer-generated and does not replace professional medical advice. "
            "For questions or concerns, please consult your healthcare provider.</i>",
            ParagraphStyle('Footer', parent=self.styles['Normal'], fontSize=9, 
                           textColor=colors.grey, alignment=TA_CENTER)
        ))
        
        # Build PDF
        doc.build(story)
        
        return output_path

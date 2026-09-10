"""Minimal pure-Python .docx (OOXML) writer: headings, paragraphs (with mono/color),
and tables. No external deps."""
import zipfile, os, html
def esc(t): return html.escape(str(t),quote=True)
class Docx:
    def __init__(self): self.body=[]
    def _runs(self,text,mono=False,bold=False,color=None,size=None):
        rpr=[]
        if bold: rpr.append("<w:b/>")
        if mono: rpr.append('<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>')
        if color: rpr.append(f'<w:color w:val="{color}"/>')
        if size: rpr.append(f'<w:sz w:val="{size*2}"/>')
        rpr='<w:rPr>'+''.join(rpr)+'</w:rPr>' if rpr else ''
        # preserve line breaks
        parts=esc(text).split("\n")
        runs=""
        for i,p in enumerate(parts):
            runs+=f'<w:r>{rpr}<w:t xml:space="preserve">{p}</w:t></w:r>'
            if i<len(parts)-1: runs+='<w:r>{}<w:br/></w:r>'.format(rpr)
        return runs
    def heading(self,text,level=1):
        sz={1:18,2:14,3:12}.get(level,12)
        self.body.append(f'<w:p><w:pPr><w:spacing w:before="240" w:after="80"/></w:pPr>{self._runs(text,bold=True,size=sz,color="1F3864")}</w:p>')
    def para(self,text,mono=False,bold=False,color=None,size=11,shade=None):
        ppr='<w:pPr>'
        if shade: ppr+=f'<w:shd w:val="clear" w:fill="{shade}"/>'
        ppr+='<w:spacing w:after="60"/></w:pPr>'
        self.body.append(f'<w:p>{ppr}{self._runs(text,mono=mono,bold=bold,color=color,size=size)}</w:p>')
    def spacer(self): self.body.append('<w:p/>')
    def table(self,rows,header=True,widths=None):
        ncol=len(rows[0]); widths=widths or [str(9000//ncol)]*ncol
        grid='<w:tblGrid>'+''.join(f'<w:gridCol w:w="{w}"/>' for w in widths)+'</w:tblGrid>'
        trs=""
        for ri,row in enumerate(rows):
            cells=""
            for ci,c in enumerate(row):
                fill=' w:fill="1F3864"' if (header and ri==0) else (' w:fill="F2F2F2"' if ri%2 else ' w:fill="FFFFFF"')
                col="FFFFFF" if (header and ri==0) else "000000"
                bold=(header and ri==0)
                cells+=(f'<w:tc><w:tcPr><w:tcW w:w="{widths[ci]}" w:type="dxa"/>'
                        f'<w:shd w:val="clear"{fill}/>'
                        '<w:tcBorders><w:top w:val="single" w:sz="4" w:color="AAAAAA"/><w:bottom w:val="single" w:sz="4" w:color="AAAAAA"/><w:left w:val="single" w:sz="4" w:color="AAAAAA"/><w:right w:val="single" w:sz="4" w:color="AAAAAA"/></w:tcBorders></w:tcPr>'
                        f'<w:p><w:pPr><w:spacing w:after="0"/></w:pPr>{self._runs(c,bold=bold,color=col,size=10)}</w:p></w:tc>')
            trs+=f'<w:tr>{cells}</w:tr>'
        self.body.append(f'<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="9000" w:type="dxa"/><w:tblBorders><w:top w:val="single" w:sz="4" w:color="AAAAAA"/><w:bottom w:val="single" w:sz="4" w:color="AAAAAA"/><w:left w:val="single" w:sz="4" w:color="AAAAAA"/><w:right w:val="single" w:sz="4" w:color="AAAAAA"/><w:insideH w:val="single" w:sz="4" w:color="AAAAAA"/><w:insideV w:val="single" w:sz="4" w:color="AAAAAA"/></w:tblBorders></w:tblPr>{grid}{trs}</w:tbl>')
        self.spacer()
    def save(self,path):
        doc=('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
             '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
             '<w:body>'+''.join(self.body)+
             '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080"/></w:sectPr>'
             '</w:body></w:document>')
        ct=('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            '</Types>')
        rels=('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
              '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
              '</Relationships>')
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600); os.close(fd)
        with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml",ct); z.writestr("_rels/.rels",rels); z.writestr("word/document.xml",doc)
        os.chmod(path,0o600)

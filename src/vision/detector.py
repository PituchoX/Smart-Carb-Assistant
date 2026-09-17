from ultralytics import YOLO

class DetetorAlimentos:
    def __init__(self, modelo_pesos: str = "yolov8s.pt"):
        self.modelo = YOLO(modelo_pesos)

    def detetar(
        self, 
        caminho_imagem: str, 
        confianca_min: float = 0.45, 
        iou_limiar: float = 0.40,
        imgsz: int = 1024
    ) -> list:
        
        resultados = self.modelo(
            caminho_imagem, 
            conf=confianca_min, 
            iou=iou_limiar, 
            imgsz=imgsz
        )[0]

        alimentos_encontrados = []
        for box in resultados.boxes:
            indice_classe = int(box.cls[0])
            nome_classe = self.modelo.names[indice_classe]
            confianca = float(box.conf[0])

            alimentos_encontrados.append({
                "classe": nome_classe,
                "confianca": round(confianca, 3)
            })

        return alimentos_encontrados
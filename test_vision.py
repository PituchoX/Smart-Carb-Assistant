from ultralytics import YOLO

modelo = YOLO("yolov8s.pt")

resultados = modelo("maça.jpg", conf=0.42, iou=0.40, imgsz=1024)

print("\n--- RESULTADO FILTRADO (YOLOV8-SMALL) ---")
alimentos = resultados[0].boxes

if len(alimentos) == 0:
    print("Nenhum alimento atingiu o limiar de confiança.")
else:
    for box in alimentos:
        classe = modelo.names[int(box.cls[0])]
        confianca = float(box.conf[0])
        print(f"- {classe}: {confianca * 100:.1f}%")

resultados[0].save(filename="resultado_limpo.jpg")
print("\nImagem anotada guardada como 'resultado_limpo.jpg'")
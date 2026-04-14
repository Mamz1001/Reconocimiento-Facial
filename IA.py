"""
=============================================================
  SISTEMA DE RECONOCIMIENTO FACIAL  |  face_recognition + Tkinter
=============================================================
  Version UI Modernizada + CORRECCIÓN DE BLOQUEOS
    - Eliminado uso de CNN (lento y se traba)
    - Redimensionamiento automático de imágenes grandes
    - Mejor manejo de errores
=============================================================
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import numpy as np
from PIL import Image, ImageTk, ImageOps
import io

# =============================================================================
#  CONFIGURACION DE RUTAS (NO MODIFICADO)
# =============================================================================

RUTA_MODELOS = r"C:\Users\Equipo 1\Desktop\Reconocimiento Facial\Reconocimiento Facial\temp_models\master\face_recognition_models-master"

MODELO_DETECTOR   = os.path.join(RUTA_MODELOS, "mmod_human_face_detector.dat")
MODELO_LANDMARKS  = os.path.join(RUTA_MODELOS, "shape_predictor_68_face_landmarks.dat")
MODELO_LANDMARKS5 = os.path.join(RUTA_MODELOS, "shape_predictor_5_face_landmarks.dat")
MODELO_RESNET     = os.path.join(RUTA_MODELOS, "dlib_face_recognition_resnet_model_v1.dat")

# Verificacion de modelos
for ruta_mod, nombre_mod in [
    (MODELO_DETECTOR,  "mmod_human_face_detector.dat"),
    (MODELO_LANDMARKS, "shape_predictor_68_face_landmarks.dat"),
    (MODELO_RESNET,    "dlib_face_recognition_resnet_model_v1.dat"),
]:
    if not os.path.isfile(ruta_mod):
        sys.exit(f"No se encontro el modelo:\n    {ruta_mod}")

# =============================================================================
#  PARCHE: Inyectamos face_recognition_models en sys.modules (NO MODIFICADO)
# =============================================================================
import types

frm = types.ModuleType("face_recognition_models")
frm.pose_predictor_model_location            = lambda: MODELO_LANDMARKS
frm.pose_predictor_five_point_model_location = lambda: MODELO_LANDMARKS5
frm.face_recognition_model_location         = lambda: MODELO_RESNET
frm.cnn_face_detector_model_location        = lambda: MODELO_DETECTOR
sys.modules["face_recognition_models"] = frm

# Importar face_recognition
try:
    import face_recognition
except ImportError:
    sys.exit("Instala face_recognition:  pip install face_recognition")

# OpenCV para visualizacion
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

PIL_AVAILABLE = True

# =============================================================================
#  FUNCIONES DE PROCESAMIENTO DE IMAGENES (CORREGIDAS)
# =============================================================================

def corregir_orientacion_imagen(ruta_imagen):
    try:
        imagen = Image.open(ruta_imagen)
        try:
            imagen = ImageOps.exif_transpose(imagen)
        except:
            pass
        if imagen.mode != 'RGB':
            imagen = imagen.convert('RGB')
        return imagen
    except Exception as e:
        print(f"Error corrigiendo orientacion {ruta_imagen}: {e}")
        return None

def redimensionar_si_necesario(imagen_pil, tamaño_maximo=1024):
    """Reduce la imagen si supera el tamaño máximo (evita lentitud)"""
    ancho, alto = imagen_pil.size
    if max(ancho, alto) > tamaño_maximo:
        factor = tamaño_maximo / max(ancho, alto)
        nuevo_ancho = int(ancho * factor)
        nuevo_alto = int(alto * factor)
        return imagen_pil.resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)
    return imagen_pil

def cargar_imagen_corregida(ruta_imagen):
    imagen_pil = corregir_orientacion_imagen(ruta_imagen)
    if imagen_pil is None:
        return None
    # Redimensionar imágenes grandes para acelerar detección
    imagen_pil = redimensionar_si_necesario(imagen_pil, tamaño_maximo=1024)
    imagen_rgb = np.array(imagen_pil)
    return imagen_rgb

def detectar_rostros_mejorado(imagen_rgb):
    """
    Solo usa HOG (rápido). Eliminado CNN porque se traba.
    Además, se prueban solo escalas razonables (0.7, 1.0, 1.3)
    """
    if imagen_rgb is None:
        return [], []
    
    # Primero intento con tamaño original
    ubicaciones = face_recognition.face_locations(imagen_rgb, model='hog')
    
    if len(ubicaciones) == 0:
        # Escalas más eficientes: reducir un poco o aumentar un poco
        for escala in [0.7, 1.3]:
            nuevo_alto = int(imagen_rgb.shape[0] * escala)
            nuevo_ancho = int(imagen_rgb.shape[1] * escala)
            # Evitar dimensiones demasiado pequeñas
            if nuevo_alto < 50 or nuevo_ancho < 50:
                continue
            imagen_escalada = cv2.resize(imagen_rgb, (nuevo_ancho, nuevo_alto))
            ubicaciones = face_recognition.face_locations(imagen_escalada, model='hog')
            if len(ubicaciones) > 0:
                # Escalar coordenadas de vuelta
                ubicaciones = [(int(top/escala), int(right/escala), 
                               int(bottom/escala), int(left/escala)) 
                               for (top, right, bottom, left) in ubicaciones]
                break
    
    embeddings = face_recognition.face_encodings(imagen_rgb, ubicaciones)
    return ubicaciones, embeddings

def redimensionar_imagen_vertical(imagen, tamaño_maximo=(300, 300)):
    if imagen is None:
        return None
    ancho_original, alto_original = imagen.size
    if ancho_original > alto_original:
        proporcion = tamaño_maximo[1] / alto_original
        nuevo_ancho = int(ancho_original * proporcion)
        nuevo_alto = tamaño_maximo[1]
        if nuevo_ancho > tamaño_maximo[0]:
            proporcion = tamaño_maximo[0] / ancho_original
            nuevo_ancho = tamaño_maximo[0]
            nuevo_alto = int(alto_original * proporcion)
    else:
        proporcion = tamaño_maximo[1] / alto_original
        nuevo_ancho = int(ancho_original * proporcion)
        nuevo_alto = tamaño_maximo[1]
        if nuevo_ancho > tamaño_maximo[0]:
            proporcion = tamaño_maximo[0] / ancho_original
            nuevo_ancho = tamaño_maximo[0]
            nuevo_alto = int(alto_original * proporcion)
    imagen_redimensionada = imagen.resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)
    lienzo = Image.new('RGB', tamaño_maximo, (18, 20, 25))
    x = (tamaño_maximo[0] - nuevo_ancho) // 2
    y = (tamaño_maximo[1] - nuevo_alto) // 2
    lienzo.paste(imagen_redimensionada, (x, y))
    return lienzo

# =============================================================================
#  FUNCIONES DE NEGOCIO (CORREGIDAS: indexado más robusto)
# =============================================================================

def indexar_fotos_multiple(carpeta, callback_progreso=None):
    base_de_datos = {}
    extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not os.path.isdir(carpeta):
        return base_de_datos
    
    personas = [p for p in os.listdir(carpeta) if os.path.isdir(os.path.join(carpeta, p))]
    
    for idx, persona in enumerate(personas):
        if callback_progreso:
            callback_progreso(f"Procesando: {persona}", idx, len(personas))
        
        ruta_persona = os.path.join(carpeta, persona)
        embeddings_lista = []
        imagenes_procesadas = 0
        errores = 0
        
        # Listar archivos de imagen
        archivos_imagen = [f for f in os.listdir(ruta_persona) 
                           if os.path.splitext(f)[1].lower() in extensiones]
        
        for archivo in archivos_imagen:
            ruta_imagen = os.path.join(ruta_persona, archivo)
            try:
                imagen_rgb = cargar_imagen_corregida(ruta_imagen)
                if imagen_rgb is not None:
                    ubicaciones, embeddings = detectar_rostros_mejorado(imagen_rgb)
                    if embeddings:
                        embeddings_lista.append(embeddings[0])
                        imagenes_procesadas += 1
                    else:
                        errores += 1
                        print(f"⚠️ Sin rostro en: {archivo} ({persona})")
                else:
                    errores += 1
            except Exception as e:
                errores += 1
                print(f"❌ Error en {archivo}: {e}")
        
        if embeddings_lista:
            embedding_promedio = np.mean(embeddings_lista, axis=0)
            base_de_datos[persona] = {
                'embedding': embedding_promedio,
                'num_muestras': len(embeddings_lista),
                'imagenes': imagenes_procesadas,
                'errores': errores
            }
            print(f"✅ {persona}: {len(embeddings_lista)} rostros de {imagenes_procesadas+errores} imágenes")
        else:
            print(f"❌ {persona}: No se detectó ningún rostro válido - se omite")
    
    return base_de_datos

def registrar_persona(nombre, lista_rutas_imagenes, carpeta_destino):
    if not nombre or not lista_rutas_imagenes:
        return False
    carpeta_persona = os.path.join(carpeta_destino, nombre)
    os.makedirs(carpeta_persona, exist_ok=True)
    import shutil
    fotos_guardadas = 0
    for i, ruta_imagen in enumerate(lista_rutas_imagenes):
        try:
            imagen_corregida = corregir_orientacion_imagen(ruta_imagen)
            if imagen_corregida:
                extension = os.path.splitext(ruta_imagen)[1]
                nombre_destino = f"foto_{i+1}{extension}"
                ruta_destino = os.path.join(carpeta_persona, nombre_destino)
                imagen_corregida.save(ruta_destino)
                fotos_guardadas += 1
        except Exception as e:
            print(f"Error guardando imagen: {e}")
    return fotos_guardadas > 0

def identificar_rostro(vector_prueba, base_de_datos, umbral=0.6):
    if not base_de_datos:
        return "Base de datos vacia", float("inf"), None
    mejor_nombre    = "Sujeto no reconocido"
    menor_distancia = float("inf")
    mejor_info = None
    for nombre, datos in base_de_datos.items():
        vector_conocido = datos['embedding']
        distancia = np.linalg.norm(vector_prueba - vector_conocido)
        if distancia < menor_distancia:
            menor_distancia = distancia
            if distancia < umbral:
                mejor_nombre = nombre
                mejor_info = datos
    return mejor_nombre, menor_distancia, mejor_info

def mostrar_con_opencv(ruta_imagen, nombre, distancia, info_persona=None):
    if not CV2_AVAILABLE:
        return
    imagen_corregida = cargar_imagen_corregida(ruta_imagen)
    if imagen_corregida is None:
        return
    imagen_bgr = cv2.cvtColor(imagen_corregida, cv2.COLOR_RGB2BGR)
    ubicaciones = face_recognition.face_locations(imagen_corregida)
    color = (0, 200, 80) if nombre != "Sujeto no reconocido" else (0, 60, 220)
    if info_persona:
        etiqueta = f"{nombre}  ({distancia:.3f})  [{info_persona['num_muestras']} muestras]"
    else:
        etiqueta = f"{nombre}  ({distancia:.3f})"
    for (top, right, bottom, left) in ubicaciones:
        cv2.rectangle(imagen_bgr, (left, top), (right, bottom), color, 2)
        cv2.rectangle(imagen_bgr, (left, bottom - 28), (right, bottom), color, cv2.FILLED)
        cv2.putText(imagen_bgr, etiqueta, (left + 6, bottom - 8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("Resultado - Reconocimiento Facial", imagen_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# =============================================================================
#  UI MODERNIZADA - Clase de estilos centralizada (SIN CAMBIOS)
# =============================================================================

class ModernStyle:
    """Clase centralizada para estilos modernos de la UI"""
    
    # Paleta de colores profesional dark mode
    BG_PRIMARY = "#0A0C10"
    BG_SECONDARY = "#14161C"
    BG_CARD = "#1A1D24"
    BG_INPUT = "#22252E"
    
    TEXT_PRIMARY = "#FFFFFF"
    TEXT_SECONDARY = "#A0A5B5"
    TEXT_DISABLED = "#5A5E6E"
    
    ACCENT_PRIMARY = "#00D4FF"
    ACCENT_SECONDARY = "#7C3AED"
    ACCENT_SUCCESS = "#10B981"
    ACCENT_DANGER = "#EF4444"
    ACCENT_WARNING = "#F59E0B"
    
    BORDER_COLOR = "#2A2D35"
    BORDER_FOCUS = "#00D4FF"
    
    # Fuentes modernas (con fallbacks)
    FONT_TITLE = ("Segoe UI", 24, "bold")
    FONT_SUBTITLE = ("Segoe UI", 11)
    FONT_HEADER = ("Segoe UI", 10, "bold")
    FONT_BODY = ("Segoe UI", 9)
    FONT_BUTTON = ("Segoe UI", 10, "bold")
    FONT_SMALL = ("Segoe UI", 8)
    
    @staticmethod
    def aplicar_sombra(widget, radius=10):
        widget.configure(relief="flat", borderwidth=0)

class ModernCard(tk.Frame):
    def __init__(self, parent, bg_color=ModernStyle.BG_CARD, **kwargs):
        super().__init__(parent, bg=bg_color, **kwargs)
        self.configure(relief="flat", borderwidth=0, highlightthickness=1, 
                      highlightbackground=ModernStyle.BORDER_COLOR, 
                      highlightcolor=ModernStyle.BORDER_COLOR)

# =============================================================================
#  GUI PRINCIPAL MODERNIZADA (SIN CAMBIOS EN LA LÓGICA DE UI)
# =============================================================================

class App(tk.Tk):
    CARPETA_FOTOS = r"C:\Users\Equipo 1\Desktop\Reconocimiento Facial\Reconocimiento Facial\Fotos"
    UMBRAL        = 0.6

    def __init__(self):
        super().__init__()
        self.title("Face Recognition System")
        
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        width = min(950, int(screen_width * 0.85))
        height = min(850, int(screen_height * 0.85))
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.minsize(850, 750)
        self.configure(bg=ModernStyle.BG_PRIMARY)
        
        self.base_de_datos = {}
        self.ruta_imagen = tk.StringVar(value="")
        self.imagenes_registro = []
        self.progress_window = None
        
        self._construir_ui()
        self._cargar_base_inicial()
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
    
    def _crear_tarjeta(self, parent, titulo, icono="", **kwargs):
        card = ModernCard(parent, padx=20, pady=15)
        card.pack(fill="x", pady=10, **kwargs)
        header_frame = tk.Frame(card, bg=ModernStyle.BG_CARD)
        header_frame.pack(fill="x", pady=(0, 12))
        tk.Label(header_frame, text=f"{icono} {titulo}", 
                bg=ModernStyle.BG_CARD, fg=ModernStyle.ACCENT_PRIMARY,
                font=ModernStyle.FONT_HEADER).pack(side="left")
        ttk.Separator(card, orient='horizontal').pack(fill="x", pady=(0, 10))
        return card
    
    def _crear_boton_moderno(self, parent, texto, comando, tipo="primary", **kwargs):
        colores = {
            "primary": (ModernStyle.ACCENT_PRIMARY, "#000000", "#00E5FF", "#000000"),
            "secondary": (ModernStyle.ACCENT_SECONDARY, "#FFFFFF", "#8B5CF6", "#FFFFFF"),
            "success": (ModernStyle.ACCENT_SUCCESS, "#FFFFFF", "#34D399", "#FFFFFF"),
            "danger": (ModernStyle.ACCENT_DANGER, "#FFFFFF", "#F87171", "#FFFFFF"),
            "dark": (ModernStyle.BG_INPUT, ModernStyle.TEXT_PRIMARY, "#2D3240", ModernStyle.TEXT_PRIMARY)
        }
        bg_color, fg_color, bg_hover, fg_hover = colores.get(tipo, colores["primary"])
        btn = tk.Button(parent, text=texto, command=comando,
                       bg=bg_color, fg=fg_color,
                       font=ModernStyle.FONT_BUTTON,
                       bd=0, relief="flat",
                       activebackground=bg_hover,
                       activeforeground=fg_hover,
                       cursor="hand2",
                       padx=15, pady=8,
                       **kwargs)
        def on_enter(e):
            btn.configure(bg=bg_hover)
        def on_leave(e):
            btn.configure(bg=bg_color)
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        return btn
    
    def _construir_ui(self):
        main_container = tk.Frame(self, bg=ModernStyle.BG_PRIMARY)
        main_container.pack(fill="both", expand=True)
        canvas = tk.Canvas(main_container, bg=ModernStyle.BG_PRIMARY, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=ModernStyle.BG_PRIMARY)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=canvas.winfo_width())
        canvas.configure(yscrollcommand=scrollbar.set)
        def _configure_canvas(event):
            canvas.itemconfig(1, width=event.width)
        canvas.bind("<Configure>", _configure_canvas)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Título
        title_frame = tk.Frame(scrollable_frame, bg=ModernStyle.BG_PRIMARY)
        title_frame.pack(fill="x", pady=(30, 10), padx=30)
        tk.Label(title_frame, text="🔍 FACE RECOGNITION", 
                bg=ModernStyle.BG_PRIMARY, fg=ModernStyle.ACCENT_PRIMARY,
                font=ModernStyle.FONT_TITLE).pack()
        tk.Label(title_frame, text="Sistema avanzado de reconocimiento facial con múltiples muestras",
                bg=ModernStyle.BG_PRIMARY, fg=ModernStyle.TEXT_SECONDARY,
                font=ModernStyle.FONT_SUBTITLE).pack(pady=(5, 0))
        
        # Tarjeta Estado
        card_estado = self._crear_tarjeta(scrollable_frame, "ESTADO DEL SISTEMA", "📊", padx=30)
        estado_content = tk.Frame(card_estado, bg=ModernStyle.BG_CARD)
        estado_content.pack(fill="x", pady=5)
        info_frame = tk.Frame(estado_content, bg=ModernStyle.BG_CARD)
        info_frame.pack(side="left", fill="x", expand=True)
        self.lbl_db = tk.Label(info_frame, text="⏳ Inicializando...",
                              bg=ModernStyle.BG_CARD, fg=ModernStyle.ACCENT_PRIMARY,
                              font=("Segoe UI", 12, "bold"))
        self.lbl_db.pack(anchor="w", pady=2)
        self.lbl_muestras = tk.Label(info_frame, text="",
                                     bg=ModernStyle.BG_CARD, fg=ModernStyle.TEXT_SECONDARY,
                                     font=ModernStyle.FONT_BODY)
        self.lbl_muestras.pack(anchor="w")
        btn_frame = tk.Frame(estado_content, bg=ModernStyle.BG_CARD)
        btn_frame.pack(side="right")
        self._crear_boton_moderno(btn_frame, "🔄 REENTRENAR MODELO", 
                                   self._generar_modelo, "primary").pack()
        
        # Tarjeta Registrar
        card_registro = self._crear_tarjeta(scrollable_frame, "REGISTRAR NUEVA PERSONA", "📸", padx=30)
        select_frame = tk.Frame(card_registro, bg=ModernStyle.BG_CARD)
        select_frame.pack(fill="x", pady=10)
        self.btn_fotos = self._crear_boton_moderno(select_frame, "📁 SELECCIONAR FOTOS", 
                                                    self._seleccionar_fotos_registro, "dark")
        self.btn_fotos.pack(side="left", padx=(0, 15))
        self.lbl_fotos_count = tk.Label(select_frame, text="0 fotos seleccionadas",
                                        bg=ModernStyle.BG_CARD, fg=ModernStyle.TEXT_SECONDARY,
                                        font=ModernStyle.FONT_BODY)
        self.lbl_fotos_count.pack(side="left", expand=True)
        self.btn_registrar = self._crear_boton_moderno(select_frame, "✅ REGISTRAR PERSONA", 
                                                        self._registrar_persona, "success")
        self.btn_registrar.pack(side="right")
        self.btn_registrar.config(state="disabled")
        preview_container = tk.Frame(card_registro, bg=ModernStyle.BG_INPUT, 
                                     height=300, width=300, relief="flat",
                                     highlightthickness=1, highlightbackground=ModernStyle.BORDER_COLOR)
        preview_container.pack(pady=15)
        preview_container.pack_propagate(False)
        self.preview_registro = tk.Label(preview_container, bg=ModernStyle.BG_INPUT)
        self.preview_registro.pack(expand=True, fill="both")
        
        # Tarjeta Probar
        card_prueba = self._crear_tarjeta(scrollable_frame, "PROBAR RECONOCIMIENTO", "🔍", padx=30)
        file_frame = tk.Frame(card_prueba, bg=ModernStyle.BG_CARD)
        file_frame.pack(fill="x", pady=10)
        self.entry_ruta = tk.Entry(file_frame, textvariable=self.ruta_imagen,
                                   bg=ModernStyle.BG_INPUT, fg=ModernStyle.TEXT_PRIMARY,
                                   font=ModernStyle.FONT_BODY, bd=0, relief="flat",
                                   insertbackground=ModernStyle.ACCENT_PRIMARY,
                                   highlightthickness=1, highlightcolor=ModernStyle.BORDER_FOCUS,
                                   highlightbackground=ModernStyle.BORDER_COLOR)
        self.entry_ruta.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 10))
        self._crear_boton_moderno(file_frame, "📂 EXPLORAR", self._seleccionar_imagen, "dark").pack(side="left")
        preview_test_container = tk.Frame(card_prueba, bg=ModernStyle.BG_INPUT, 
                                          height=300, width=300, relief="flat",
                                          highlightthickness=1, highlightbackground=ModernStyle.BORDER_COLOR)
        preview_test_container.pack(pady=15)
        preview_test_container.pack_propagate(False)
        self.preview_prueba = tk.Label(preview_test_container, bg=ModernStyle.BG_INPUT)
        self.preview_prueba.pack(expand=True, fill="both")
        self.btn_analizar = self._crear_boton_moderno(card_prueba, "🔍 ANALIZAR ROSTRO", 
                                                       self._analizar, "secondary")
        self.btn_analizar.pack(fill="x", pady=10)
        resultado_container = tk.Frame(card_prueba, bg=ModernStyle.BG_INPUT,
                                       relief="flat", highlightthickness=1,
                                       highlightbackground=ModernStyle.BORDER_COLOR)
        resultado_container.pack(fill="x", pady=10)
        resultado_inner = tk.Frame(resultado_container, bg=ModernStyle.BG_INPUT, padx=15, pady=12)
        resultado_inner.pack(fill="x")
        tk.Label(resultado_inner, text="RESULTADO",
                bg=ModernStyle.BG_INPUT, fg=ModernStyle.TEXT_SECONDARY,
                font=ModernStyle.FONT_HEADER).pack(anchor="w")
        self.lbl_resultado = tk.Label(resultado_inner, text="⏳ Esperando imagen...",
                                      bg=ModernStyle.BG_INPUT, fg=ModernStyle.TEXT_PRIMARY,
                                      font=("Segoe UI", 14, "bold"), wraplength=500)
        self.lbl_resultado.pack(anchor="w", pady=(8, 4))
        self.lbl_distancia = tk.Label(resultado_inner, text="",
                                      bg=ModernStyle.BG_INPUT, fg=ModernStyle.TEXT_SECONDARY,
                                      font=ModernStyle.FONT_SMALL)
        self.lbl_distancia.pack(anchor="w")
        self.lbl_info = tk.Label(resultado_inner, text="",
                                 bg=ModernStyle.BG_INPUT, fg=ModernStyle.ACCENT_SUCCESS,
                                 font=ModernStyle.FONT_SMALL)
        self.lbl_info.pack(anchor="w", pady=(4, 0))
        tk.Frame(scrollable_frame, bg=ModernStyle.BG_PRIMARY, height=30).pack()
    
    def _mostrar_progreso(self, mensaje, actual, total):
        if self.progress_window is None or not self.progress_window.winfo_exists():
            self.progress_window = tk.Toplevel(self)
            self.progress_window.title("Generando Modelo")
            self.progress_window.geometry("450x160")
            self.progress_window.configure(bg=ModernStyle.BG_CARD)
            self.progress_window.transient(self)
            self.progress_window.grab_set()
            tk.Label(self.progress_window, text="🔄 Procesando imágenes...",
                    bg=ModernStyle.BG_CARD, fg=ModernStyle.ACCENT_PRIMARY,
                    font=("Segoe UI", 12, "bold")).pack(pady=20)
            self.progress_bar = ttk.Progressbar(self.progress_window, length=350, mode='determinate')
            self.progress_bar.pack(pady=10)
            self.progress_label = tk.Label(self.progress_window, text="",
                                          bg=ModernStyle.BG_CARD, fg=ModernStyle.TEXT_SECONDARY,
                                          font=ModernStyle.FONT_BODY)
            self.progress_label.pack(pady=10)
        self.progress_bar['maximum'] = total
        self.progress_bar['value'] = actual
        self.progress_label.config(text=mensaje)
        self.progress_window.update_idletasks()  # Forzar actualización
        if actual >= total:
            self.progress_window.after(1000, self.progress_window.destroy)
            self.progress_window = None

    def _cargar_base_inicial(self):
        threading.Thread(target=self._generar_modelo, daemon=True).start()

    def _generar_modelo(self):
        self.lbl_db.config(text="⏳ Generando modelo...", fg=ModernStyle.ACCENT_PRIMARY)
        self.lbl_muestras.config(text="")
        def update_progress(mensaje, actual, total):
            self.after(0, lambda: self._mostrar_progreso(mensaje, actual, total))
        base = indexar_fotos_multiple(self.CARPETA_FOTOS, update_progress)
        self.base_de_datos = base
        def actualizar_ui():
            if self.base_de_datos:
                total_muestras = sum(d['num_muestras'] for d in self.base_de_datos.values())
                self.lbl_db.config(text=f"✅ {len(self.base_de_datos)} persona(s) registrada(s)",
                                  fg=ModernStyle.ACCENT_SUCCESS)
                self.lbl_muestras.config(text=f"📊 {total_muestras} muestras de rostros en total",
                                        fg=ModernStyle.TEXT_SECONDARY)
            else:
                self.lbl_db.config(text="⚠️ Sin personas registradas", fg=ModernStyle.ACCENT_WARNING)
                self.lbl_muestras.config(text="💡 Usa 'Registrar Nueva Persona' para comenzar",
                                        fg=ModernStyle.TEXT_SECONDARY)
        self.after(0, actualizar_ui)

    def _seleccionar_fotos_registro(self):
        rutas = filedialog.askopenfilenames(
            title="Selecciona FOTOS de la misma persona",
            filetypes=[("Imagenes", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if rutas:
            self.imagenes_registro = list(rutas)
            self.lbl_fotos_count.config(text=f"📸 {len(rutas)} foto(s) seleccionada(s)",
                                       fg=ModernStyle.ACCENT_SUCCESS)
            self.btn_registrar.config(state="normal")
            if PIL_AVAILABLE:
                try:
                    imagen = corregir_orientacion_imagen(rutas[0])
                    if imagen:
                        imagen_redim = redimensionar_imagen_vertical(imagen, (280, 280))
                        photo = ImageTk.PhotoImage(imagen_redim)
                        self.preview_registro.config(image=photo, text="")
                        self.preview_registro.image = photo
                except Exception as e:
                    print(f"Error en preview: {e}")
        else:
            self.imagenes_registro = []
            self.lbl_fotos_count.config(text="0 fotos seleccionadas", fg=ModernStyle.TEXT_SECONDARY)
            self.btn_registrar.config(state="disabled")
            self.preview_registro.config(image="", text="")

    def _registrar_persona(self):
        if not self.imagenes_registro:
            messagebox.showwarning("Sin fotos", "Selecciona al menos una foto primero.")
            return
        nombre = simpledialog.askstring("Nombre", "Ingresa el nombre de la persona:",
                                        parent=self)
        if not nombre or not nombre.strip():
            return
        nombre = nombre.strip()
        if nombre in self.base_de_datos:
            respuesta = messagebox.askyesno(
                "Persona existente",
                f"'{nombre}' ya está registrada.\n¿Deseas agregar estas fotos a su perfil?"
            )
            if not respuesta:
                return
        if registrar_persona(nombre, self.imagenes_registro, self.CARPETA_FOTOS):
            messagebox.showinfo("Éxito", 
                               f"✅ Persona '{nombre}' registrada con {len(self.imagenes_registro)} foto(s)\n"
                               f"🔄 Reentrenando modelo...")
            self.imagenes_registro = []
            self.lbl_fotos_count.config(text="0 fotos seleccionadas", fg=ModernStyle.TEXT_SECONDARY)
            self.btn_registrar.config(state="disabled")
            self.preview_registro.config(image="")
            threading.Thread(target=self._generar_modelo, daemon=True).start()
        else:
            messagebox.showerror("Error", "No se pudo registrar la persona.\nVerifica que las fotos tengan rostros visibles.")

    def _seleccionar_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Selecciona la imagen de prueba",
            filetypes=[("Imagenes", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if ruta:
            self.ruta_imagen.set(ruta)
            if PIL_AVAILABLE:
                try:
                    imagen = corregir_orientacion_imagen(ruta)
                    if imagen:
                        imagen_redim = redimensionar_imagen_vertical(imagen, (280, 280))
                        photo = ImageTk.PhotoImage(imagen_redim)
                        self.preview_prueba.config(image=photo, text="")
                        self.preview_prueba.image = photo
                except Exception as e:
                    print(f"Error en preview: {e}")

    def _analizar(self):
        ruta = self.ruta_imagen.get().strip()
        if not ruta or not os.path.isfile(ruta):
            messagebox.showwarning("Sin imagen", "Selecciona una imagen de prueba válida.")
            return
        if not self.base_de_datos:
            messagebox.showerror("Base vacía", 
                               "No hay personas registradas.\n\nUsa 'Registrar Nueva Persona' primero.")
            return
        self.btn_analizar.config(text="⏳ Procesando...", state="disabled")
        threading.Thread(target=self._proceso_analisis, args=(ruta,), daemon=True).start()

    def _proceso_analisis(self, ruta):
        try:
            imagen_rgb = cargar_imagen_corregida(ruta)
            if imagen_rgb is None:
                self.after(0, lambda: self._mostrar_error("No se pudo cargar la imagen"))
                return
            ubicaciones, embeddings = detectar_rostros_mejorado(imagen_rgb)
            if not embeddings:
                self.after(0, self._mostrar_sin_rostro)
                return
            vector_prueba = embeddings[0]
            nombre, distancia, info = identificar_rostro(vector_prueba, self.base_de_datos, self.UMBRAL)
            self.after(0, lambda: self._mostrar_resultado(nombre, distancia, ruta, info))
            if CV2_AVAILABLE:
                mostrar_con_opencv(ruta, nombre, distancia, info)
        except Exception as e:
            self.after(0, lambda: self._mostrar_error(str(e)))
        finally:
            self.after(0, lambda: self.btn_analizar.config(text="🔍 ANALIZAR ROSTRO", state="normal"))

    def _mostrar_sin_rostro(self):
        self.lbl_resultado.config(text="❌ No se detectó ningún rostro en la imagen", 
                                  fg=ModernStyle.ACCENT_DANGER)
        self.lbl_distancia.config(text="💡 Consejo: Asegúrate que el rostro sea visible y la imagen no esté rotada",
                                 fg=ModernStyle.TEXT_SECONDARY)
        self.lbl_info.config(text="")

    def _mostrar_resultado(self, nombre, distancia, ruta, info):
        reconocido = nombre != "Sujeto no reconocido"
        color = ModernStyle.ACCENT_SUCCESS if reconocido else ModernStyle.ACCENT_DANGER
        icono = "✅" if reconocido else "❌"
        self.lbl_resultado.config(text=f"{icono}  {nombre}", fg=color)
        self.lbl_distancia.config(text=f"📏 Distancia: {distancia:.4f}  (umbral: {self.UMBRAL})",
                                 fg=ModernStyle.TEXT_SECONDARY)
        if info:
            self.lbl_info.config(text=f"📊 Basado en {info['num_muestras']} muestra(s) de rostro",
                                fg=ModernStyle.ACCENT_SUCCESS)
        else:
            self.lbl_info.config(text="")

    def _mostrar_error(self, error):
        self.lbl_resultado.config(text=f"⚠️ Error: {error}", fg=ModernStyle.ACCENT_DANGER)
        self.lbl_distancia.config(text="")
        self.lbl_info.config(text="")

# =============================================================================
if __name__ == "__main__":
    os.makedirs(App.CARPETA_FOTOS, exist_ok=True)
    app = App()
    app.mainloop()
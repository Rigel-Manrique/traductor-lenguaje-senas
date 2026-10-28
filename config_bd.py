import sqlite3

NOMBRE_BD = "sistema_senas.db"

def inicializar_bd():
    conn = sqlite3.connect(NOMBRE_BD)
    cursor = conn.cursor()

    # Tabla de Usuarios/Dispositivos
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS usuarios (
            id_usuario INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre_usuario TEXT NOT NULL,
            dispositivo TEXT NOT NULL
        )
    ''')

    # Tabla de Historial para BI
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS historial_traducciones (
            id_evento INTEGER PRIMARY KEY AUTOINCREMENT,
            id_usuario INTEGER,
            sena_detectada TEXT NOT NULL,
            confianza REAL NOT NULL,
            latencia_ms REAL NOT NULL,
            fecha_hora TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (id_usuario) REFERENCES usuarios (id_usuario)
        )
    ''')

    # Insertar usuario por defecto si la tabla está vacía
    cursor.execute("SELECT COUNT(*) FROM usuarios")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO usuarios (nombre_usuario, dispositivo) VALUES ('Usuario_Prototipo', 'PC_Local_GPU_Integrada')")

    conn.commit()
    conn.close()

def registrar_evento_traduccion(sena, confianza, latencia_ms, id_usuario=1):
    conn = sqlite3.connect(NOMBRE_BD)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO historial_traducciones (id_usuario, sena_detectada, confianza, latencia_ms)
        VALUES (?, ?, ?, ?)
    ''', (id_usuario, sena, confianza, latencia_ms))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    inicializar_bd()
    print(f"Base de datos '{NOMBRE_BD}' inicializada correctamente.")
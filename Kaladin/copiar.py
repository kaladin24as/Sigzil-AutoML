# Tomas Farias
# Modulo para copiar a portapapeles

import pyperclip

def copiar_a_portapeles(texto):
    pyperclip.copy(texto)
    print("Texto copiado al portapapeles:", texto)
from mostrarmenus import *
from machinelarning import *
from copiar import *
def main():
    while True:
        mostrar_menu()
        opcion_principal = input("Select a option: ")
        machine_learning(opcion_principal)
#power_bi(opcion_principal)
#hacking(opcion_principal)
#ayuda(opcion_principal)
#creditos(opcion_principal)

        if opcion_principal == "6":
            print("Saliendo del programa...")
            break
        else:
            print("Opcion no valida.")
            
if __name__ == "__main__":
    main()
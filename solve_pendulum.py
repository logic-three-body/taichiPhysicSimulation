import taichi as ti
ti.init(arch=ti.cpu)

def main():
    gui=ti.GUI('Pendulum System')
    while True:
        gui.show()


if __name__ == '__main__':
        main()
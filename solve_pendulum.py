import taichi as ti
ti.init(arch=ti.cpu)

def main():
    gui=ti.GUI('Pendulum System',background_color=0xfaf3e0)
    while True:
        gui.show()


if __name__ == '__main__':
        main()
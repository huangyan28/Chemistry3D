import numpy as np

class material:
    def __init__(self, name, amount, color) -> None:
        self.name = name
        self.amount = amount
        self.standard_color = color
        
    
class reaction_simple:
    def __init__(self, reactants:list, products:list, del_amount=0.01) -> None:
        self.reactants = reactants
        self.products = products
        self.del_amount = del_amount
        
    def react(self, step, speed):
        self.del_amount = self.del_amount * speed
        for i in range(step):
            for reactant in self.reactants:
                reactant.amount -= self.del_amount
            for product in self.products:
                product.amount += self.del_amount
    
    def render(self):
        pass
        
def color_dilution(rgba, org_conc, org_volume, add_solvent):

    K_const = 255./(255-100)
    add_conc = org_conc * org_volume / (org_volume + add_solvent)
    print(10 ** (-(add_conc - org_conc)))
    rgba[3] = int((1 - K_const * 10 ** (-(add_conc - org_conc)) * (1 - rgba[3]/255)) * 255)

    return rgba

def mix_solutions(rgba1, rgba2):
    rgba1, rgba2 = np.array(rgba1)/255, np.array(rgba2)/255
    a = 1 - (1 - rgba1[3]) * (1 - rgba2[3])
    r = ((rgba1[3] * rgba1[0] + (1 - rgba1[3]) * rgba2[3] * rgba2[0]) + (rgba2[3] * rgba2[0] + (1 - rgba2[3]) * rgba1[3] * rgba1[0]))/(2*a)
    g = ((rgba1[3] * rgba1[1] + (1 - rgba1[3]) * rgba2[3] * rgba2[1]) + (rgba2[3] * rgba2[1] + (1 - rgba2[3]) * rgba1[3] * rgba1[1]))/(2*a)
    b = ((rgba1[3] * rgba1[2] + (1 - rgba1[3]) * rgba2[3] * rgba2[2]) + (rgba2[3] * rgba2[2] + (1 - rgba2[3]) * rgba1[3] * rgba1[2]))/(2*a)
    r,g,b,a = int(r*255), int(g*255), int(b*255), int(a*255)
    return [r,g,b,a]
if __name__ == '__main__':
    try:
        import sys, pygame
    except:
        "No pygame module found!"
    pygame.init()
    size = width, height = (800, 800)
    screen = pygame.display.set_mode(size)
    background=pygame.image.load('1.png')
    screen.blit(background,(0,0))
    pygame.display.set_caption('reaction')
    vessel = pygame.draw.rect(screen, (130,130,130), (350-2, 200-2, 100+4, 400+4),2)
    
    # color = (255,255,255,0)
    # color = (135,209,192,100) # FeSO4
    # color = (102,10,96,100) # KMnO4
    # color = [174,119,0,100] # Fe2(SO4)3
    # color = [216,183,185,130] # MnSO4

    color1 = [135,209,192,100]
    color2 = [216,183,185,150]

    # dilution
    # color = color_dilution(color, 1, 1, 1)

    # mix
    color = mix_solutions(rgba1=color1,rgba2=color2)
    # print(color)

    surface = pygame.surface.Surface((100,350))
    surface.set_alpha(color[3])
    rect = pygame.draw.rect(surface, color, (0, 0, 100, 350))
    screen.blit(surface,(350,250))
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        
        pygame.display.update()
    
    # import cv2
    # import numpy as np

    # img = cv2.imread("1.png")

    # # 绘制半透明矩形框
    # rectangle = np.zeros((400, 100, 3), dtype=np.uint8)
    # cv2.rectangle(rectangle, (0, 0), (399, 99), (255, 255, 255), -1)
    # alpha = 0.6
    # result = cv2.addWeighted(img[200:600, 450:550], alpha, rectangle, 1 - alpha, 0)
    # img[200:600, 450:550] = result

    # cv2.imshow("Result", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

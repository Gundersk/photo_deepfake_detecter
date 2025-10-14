class Point:
    color = 'red'
    r = 1

    def printPr(self):
        print(self.color)
        print(self.r)

#print(Point.__dict__)
x = Point()
x.printPr()
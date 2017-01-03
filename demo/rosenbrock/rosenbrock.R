t <- read.table("rosenbrock.table", header=FALSE)

a <- 1
b <- 100
f <- function(x1, x2) (a - x1)^2 + b*(x2 - x1^2)^2

x <- seq(-1.5, 2, length=200)
y <- seq(-0.5, 3, length=200)
z <- outer(x, y, f)

png("rosenbrock.png")
image(x, y, z, col=heat.colors(200)[40:200])
contour(x, y, z, add=T, col="white", nlevels=20)
points(t, type="b")
dev.off()

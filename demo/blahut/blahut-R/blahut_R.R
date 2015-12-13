
# ------------------------------------------------------------------------------

binary <- "./blahut-R"
lambda <- 1.0

# ------------------------------------------------------------------------------

channel <- matrix(
    c(0.60, 0.30, 0.10,
      0.70, 0.10, 0.20,
      0.50, 0.05, 0.45),
    3, 3)

px.init <- c(1/3, 1/3, 1/3)

# ------------------------------------------------------------------------------

channel.str <- capture.output(str(channel))
px.init.str <- capture.output(str(px.init))

command <- sprintf("%s -l %f '%s' '%s'", binary, lambda, channel.str, px.init.str)
px <- eval(parse(text = system(command, intern=TRUE)))

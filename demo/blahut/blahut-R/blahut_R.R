
# Usage:
# source("PATH/blahut_R.R", chdir = TRUE)

# ------------------------------------------------------------------------------

blahut.dir <- getwd()
blahut <- function(channel, px.init, lambda = 1.0, binary = sprintf("%s/blahut-R", blahut.dir)) {
    channel.str <- capture.output(str(channel))
    px.init.str <- capture.output(str(px.init))

    command <- sprintf("%s -l %f '%s' '%s'", binary, lambda, channel.str, px.init.str)
    eval(parse(text = system(command, intern=TRUE)))
}

# ------------------------------------------------------------------------------
if (FALSE) {

    channel <- matrix(
        c(0.60, 0.30, 0.10,
          0.70, 0.10, 0.20,
          0.50, 0.05, 0.45),
        3, 3)

    px.init <- c(1/3, 1/3, 1/3)

    px <- blahut(channel, px.init)

}

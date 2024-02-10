train <- read.csv("data/pca_df_train.csv")
train
test <- read.csv("data/pca_df_test.csv")

X <- train[, 1:37]
y <- train[, 38]

genres_dict <- list(
  "thriller/horror" = 1,
  "action" = 2,
  "drama" = 3,
  "comedy" = 4,
  "family" = 5,
  "music" = 6,
  "documentary" = 7,
  "live" = 8,
  "police" = 9,
  "short" = 10
)

y <- unlist(lapply(y, function(x) genres_dict[[x]]))
y <- as.factor(y)

vcr.train <- vcr.da.train(X, y, rule = "QDA")


labels <- c("t/h", "action", "drama", "comedy", "family", "music", "doc", "live", "police", "short")
cols <- c("red", "blue", "green", "orange", "purple", "pink", "brown", "cyan", "yellow", "gray")

plot_classmap <- function(int, str, cols, labels) {
  classmap(vcr.train, int, classCols = cols, main = str, cex = 0.5)
  legend("right", fill = cols, legend = labels,
         cex = 0.7, ncol = 1, bg = "white")
}


for (i in names(genres_dict)) {
  value <- genres_dict[[i]]
  plot_classmap(value, i, cols, labels)
}

library(lubridate)
library(magrittr)
library(tidyverse)

set.seed(0)

#---------------------------
cat("Defining auxiliary functions...\n")

has_many_values <- function(x) n_distinct(x) > 1

is_na_val <- function(x) is.infinite(x) || is.nan(x)

#---------------------------
cat("Preprocessing historical transactions...\n")

htrans <- read_csv("data/historical_transactions.csv") %>% 
  rename(card = card_id)

sum_htrans_id <- htrans %>%
  group_by(card) %>%
  summarise_at(vars(ends_with("_id")), n_distinct, na.rm = TRUE) 

ohe_htrans <- htrans %>%
  select(authorized_flag, starts_with("category")) %>% 
  mutate_all(factor) %>% 
  model.matrix.lm(~ . - 1, ., na.action = NULL) %>% 
  as_tibble()

fn <- funs(mean, sd, min, max, sum, n_distinct, .args = list(na.rm = TRUE))
sum_htrans <- htrans %>%
  select(-authorized_flag, -starts_with("category"), -ends_with("_id")) %>% 
  add_count(card) %>%
  group_by(card) %>%
  mutate(date_diff = as.integer(diff(range(purchase_date))),
         prop = n() / sum(n)) %>% 
  ungroup() %>% 
  mutate(year = year(purchase_date),
         month = month(purchase_date),
         day = day(purchase_date),
         hour = hour(purchase_date)) %>% 
  select(-purchase_date) %>% 
  bind_cols(ohe_htrans) %>% 
  group_by(card) %>%
  summarise_all(fn) %>% 
  left_join(sum_htrans_id)

rm(htrans, sum_htrans_id, ohe_htrans); gc()

#---------------------------
cat("Preprocessing new transactions...\n")

ntrans <- read_csv("data/new_merchant_transactions.csv") %>% 
  left_join(read_csv("data/merchants.csv"),
            by = "merchant_id", suffix = c("", "_y")) %>%
  select(-authorized_flag) %>% 
  rename(card = card_id)

sum_ntrans_id <- ntrans %>%
  group_by(card) %>%
  summarise_at(vars(contains("_id")), n_distinct, na.rm = TRUE) 

ohe_ntrans <- ntrans %>%
  select(starts_with("category"), starts_with("most_recent")) %>% 
  mutate_all(factor) %>% 
  model.matrix.lm(~ . - 1, ., na.action = NULL) %>% 
  as_tibble()

fn <- funs(mean, sd, min, max, sum, n_distinct, .args = list(na.rm = TRUE))
sum_ntrans <- ntrans %>%
  select(-starts_with("category"), -starts_with("most_recent"), -contains("_id")) %>% 
  add_count(card) %>%
  group_by(card) %>%
  mutate(date_diff = as.integer(diff(range(purchase_date))),
         prop = n() / sum(n)) %>% 
  ungroup() %>% 
  mutate(year = year(purchase_date),
         month = month(purchase_date),
         day = day(purchase_date),
         hour = hour(purchase_date)) %>% 
  select(-purchase_date) %>% 
  bind_cols(ohe_ntrans) %>% 
  group_by(card) %>%
  summarise_all(fn) %>% 
  left_join(sum_ntrans_id)

rm(ntrans, sum_ntrans_id, ohe_ntrans, fn); gc()

#---------------------------
cat("Joining datasets...\n")

tr <- read_csv("data/train.csv") 
te <- read_csv("data/test.csv")

tri <- 1:nrow(tr)
y <- tr$target

tr_te <- tr %>% 
  select(-target) %>% 
  bind_rows(te) %>%
  rename(card = card_id) %>% 
  mutate(first_active_month = ymd(first_active_month, truncated = 1),
         year = year(first_active_month),
         month = month(first_active_month),
         date_diff = as.integer(ymd("2018-02-01") - first_active_month)) %>% 
  select(-first_active_month) %>% 
  left_join(sum_htrans, by = "card") %>% 
  left_join(sum_ntrans, by = "card") %>% 
  select(-card)

browser()
tr <- tr_te[tri, ]
te <- tr_te[-tri, ]
tr$target <- y
write_csv(tr, "data/elo_train.csv")
write_csv(te, "data/elo_test.csv")

install.packages("survival")
library(survival)
install.packages("knitr")
install.packages("kableExtra")
library(knitr)
library(kableExtra)
install.packages("tidyverse")
library(tidyverse)
?lung
data = lung
data <- lung %>% 
  mutate(sex=factor(sex, levels=c(1,2), labels=c("M", "F")),
         had_event=ifelse(status==1, 0, 1))

#na_fit <- survfit(Surv(time/365.25, had_event) ~ 1, data = data)
na_fit <- survfit(Surv(age, had_event) ~ 1, data = data)
plot(na_fit, fun = "cumhaz", xlab = "Time", ylab = "Cumulative Hazard")
na_data <- data.frame(time = na_fit$time, 
                      cumhaz = na_fit$cumhaz)
lin_reg <- lm(cumhaz ~ time, data = na_data)
reg_data <- data.frame(time = na_data$time,
                       reg_cumhaz = predict(lin_reg, newdata = na_data))

g <- ggplot(na_data, aes(time, cumhaz)) +
  geom_step() +
  xlab("Years") +
  ylab("Cumulative Hazard")


g + geom_line(data = reg_data, aes(time, reg_cumhaz), color = "red")
lin_reg_summary <- summary(lin_reg)

coefficients <- lin_reg_summary$coefficients

intercept <- coefficients["(Intercept)", "Estimate"]
slope <- coefficients["time", "Estimate"]

print(paste("Intercept:", intercept))
print(paste("Slope:", slope))

install.packages("psfmi")
library(psfmi)
data = smoking
smokers <- data %>% filter(smoking == 1)
na_fit <- survfit(Surv(time, death) ~ 1, data = smokers)
plot(na_fit, fun = "cumhaz", xlab = "Time", ylab = "Cumulative Hazard")
na_data <- data.frame(time = na_fit$time, 
                      cumhaz = na_fit$cumhaz)
lin_reg <- lm(cumhaz ~ time, data = na_data)
reg_data <- data.frame(time = na_data$time,
                       reg_cumhaz = predict(lin_reg, newdata = na_data))

g <- ggplot(na_data, aes(time, cumhaz)) +
  geom_step() +
  xlab("Years") +
  ylab("Cumulative Hazard")


g + geom_line(data = reg_data, aes(time, reg_cumhaz), color = "red")
lin_reg_summary <- summary(lin_reg)

coefficients <- lin_reg_summary$coefficients

intercept <- coefficients["(Intercept)", "Estimate"]
slope <- coefficients["time", "Estimate"]

print(paste("Intercept:", intercept))
print(paste("Slope:", slope))

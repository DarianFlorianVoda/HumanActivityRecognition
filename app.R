#install.packages("shinydashboard")
library(mosaic)
library(plotly)
library(GGally)
library(dplyr)
library(rpart)
library(caret)
library(psych)
library(ggplot2)
library(ggcorrplot)
library(rela)
library(shiny)
library(DT)
library(plotly)
library(caret)
library(isotree)

### 1. Read all samples and combine them

# Delete all variables
rm( list = ls() )
read_idle = read.csv("C:/Users/ahmed/Desktop/APPDS_Masters/Intro to ML/IML Project2/01_Idle.csv")
idle_data <- data.frame(read_idle)

read_run = read.csv("C:/Users/ahmed/Desktop/APPDS_Masters/Intro to ML/IML Project2/02_Running.csv")
run_data <- data.frame(read_run)

read_dab = read.csv("C:/Users/ahmed/Desktop/APPDS_Masters/Intro to ML/IML Project2/03_Dab.csv")
dab_data <- data.frame(read_dab)

read_siu = read.csv("C:/Users/ahmed/Desktop/APPDS_Masters/Intro to ML/IML Project2/04_Siu.csv")
siu_data <- data.frame(read_siu)

# Rename ID correctly:
names(idle_data)[1] <- "ID"
names(run_data)[1] <- "ID"
names(dab_data)[1] <- "ID"
names(siu_data)[1] <- "ID"

#### Combined data

# Overall in total there are 8985 rows
# So in the dab data for Orientation.X and Orientation.Z we have the wrong data type. <chr> instead of <dbl>
# Basically we can't scale before converting to numeric


idle_run <- rbind(idle_data, run_data)
irun_dab <- rbind(idle_run, dab_data)
motion_data <- rbind(irun_dab, siu_data)

idle_run$Orientation.X <- as.numeric(idle_run$Orientation.X)
colSums(is.na(idle_run))


### 2. Do some Exploratory Data Analysis (EDA) on whole data:
motion_data_all <- data.frame(motion_data)
# Remove Magnetic, because there are many NA's in it
motion_data_all <- motion_data_all[,!names(motion_data_all) %in% c("MagneticField.X")]
motion_data_all <- motion_data_all[,!names(motion_data_all) %in% c("MagneticField.Y")]
motion_data_all <- motion_data_all[,!names(motion_data_all) %in% c("MagneticField.Z")]

# Convert columns to correct type
motion_data_all$Category <- as.factor(motion_data_all$Category)
motion_data_all$Acceleration.X <- as.numeric(motion_data_all$Acceleration.X)
motion_data_all$Orientation.X <- as.numeric(motion_data_all$Orientation.X)
motion_data_all$Orientation.Z <- as.numeric(motion_data_all$Orientation.Z)

# More NA's found after convertion
colSums(is.na(motion_data_all))

# Remove the NA's
# About 8584 rows left

motion_data_all <- na.omit(motion_data_all)
colSums(is.na(motion_data_all))

# Scale the data:

quant_var <- select(motion_data_all, c(6:14))
cat_var <- select(motion_data_all, c(3))

quant_var <- scale(quant_var)
motion_data_scale <- cbind(cat_var, quant_var)
motion_data_scale


# Train with data from Ahmed, Tobias, Saghar and Ronaldo
#motion_data_part <- subset(motion_data_all, Author == "Ahmed" | Author == "Tobias" | Author == "Saghar" | Author == "Ronaldo") #+ subset(motion_data_all, Author == "Tobias")
#motion_data_unknown <- subset(motion_data, Author == "Regan" | Author == "Darian") # 33 %

motion_data_part <- subset(motion_data_all, Author == "Ahmed" | Author == "Darian" | Author == "Regan" | Author == "Ronaldo" | Author == "Tobias") 
motion_data_unknown <- subset(motion_data_all, Author == "Saghar" & Category == "Idle" & Sample == "idle_31") # 55.76 %

# For statistics
motion_data_all_stat <- data.frame(motion_data_all)
# Remove unrelevant columns
motion_data_all <- motion_data_all[,!names(motion_data_all) %in% c("ID", "Acceleration.Timestamp", "Author", "Sample")]

#Write merged cleaned data to file:

write.csv(motion_data_all, "All Samples Clean.csv", row.names = FALSE)

#### Category data distribution

# Stacked bar chart:
# Seems like Darian and Ahmed have compared to the others more motion data

cat_count <- group_by(motion_data_all_stat, Author, Category) %>%
  summarize(count=n())
#----------
motion_data_all_numeric <- data.frame(motion_data_all)
motion_data_all_numeric <- motion_data_all_numeric[,!names(motion_data_all_numeric) %in% c("Category")]

#Was for only for testing -> Darian: Everyone has different position of phone, thats why we should skip Orientation
#motion_data_all_numeric <- motion_data_all_numeric[,!names(motion_data_all_numeric) %in% c("Orientation.X", "Orientation.Y", "Orientation.Z")]
#motion_data_all_numeric$Category <- as.numeric(factor(motion_data_all_numeric$Category))
#motion_data_all_numeric$Category <- as.factor(motion_data_all_numeric$Category)
#-------
#We use only relevant columns for the model training

remove_col <- c("ID",  "Acceleration.Timestamp", "Author", "Sample", "Orientation.X", "Orientation.Y", "Orientation.Z")

motion_data_part <- motion_data_part[,!names(motion_data_part) %in% remove_col]
motion_data_unknown <- motion_data_unknown[,!names(motion_data_unknown) %in% remove_col]

# Calculate the correlation matrix of the data frame
cor_matrix <- cor(motion_data_all_numeric)
#-----------------
### 3. Train on whole data with selected features:
 
# Train split: 80 %, Test split: 20 %
#   
# Since the features that we selected correlate good and are relevant, we skip the angular velocity

set.seed(10)

# Take variables from correlation analysis
feature_selection <- motion_data_part#[,c("Category", "Acceleration.X", "Acceleration.Y", "Acceleration.Z")]

train_index_all <- createDataPartition(feature_selection$Category, p =0.80, list = FALSE)
train_data_all<-feature_selection[train_index_all, ]
test_data_all<-feature_selection[-train_index_all, ]

#### Accuracy on train data with rf: 86.91 % without orientation

set.seed(6)
# 6: 89.8 %
control_par <- trainControl(method = "cv", number=4)
model_rf_all <- train(Category~.,
                      data=train_data_all, 
                      "rf",
                      trControl = control_par
)

model_rf_all


#### Accuracy on testing data with rf and cv: 86.33 % without orientation


set.seed(6)
## Generate predictions
rf_all_pred_test <- predict(model_rf_all,test_data_all) 

## Print the accuracy
accuracy <- mean(rf_all_pred_test == test_data_all$Category)*100
accuracy

cm_test_data <- confusionMatrix(rf_all_pred_test, test_data_all$Category)
cm_test_data


##############
# All charts #
##############
library(dplyr)
library(shiny)
library(shinydashboard)
library(ggplot2)
library(reshape2)

server <- function(input, output){
  
  
  
  # State
  output$state_plot <- renderPlot({motion_data_box <- select(motion_data_scale, c("Acceleration.X","Acceleration.Y","Acceleration.Z","AngularVelocity.X","AngularVelocity.Y","AngularVelocity.Z","Orientation.X","Orientation.Y","Orientation.Z"))
  boxplot(motion_data_box) + scale_x_discrete(guide = guide_axis(angle = 90))
  })
  # Predictions
  output$pred_plot <- renderPlot({# Calculate the correlation matrix of the data frame
    cor_matrix <- cor(motion_data_all_numeric)
    
    # Visualize the correlation matrix using ggcorrplot
    ggcorrplot(cor_matrix, hc.order = TRUE, type = "lower", 
               lab = TRUE, lab_size = 3, method = "circle")
  })
  # Confusion Matrix
  output$CM_plot <- renderPlot({
    quantitative_variables <- test_data_all %>% select_if(is.numeric)
    ggpairs(quantitative_variables, aes(color = test_data_all$Category))
  })
  
  
  # Volume
  output$vol_plot <- renderPlot({
    
    ggplot(cat_count, aes(x = Author, y = count, fill = Category)) +geom_bar(stat = "identity")
  })
  
  # Confusion Matrix
  output$Confusion_matrix_plot <- renderPlotly({
    plt <- as.data.frame(cm_test_data$table)
    plt$Prediction <- factor(plt$Prediction, levels=rev(levels(plt$Prediction)))
    rf_conf_mat <- ggplot(plt, aes(Prediction,Reference, fill= Freq)) +
      geom_tile() + geom_text(aes(label=Freq)) +
      scale_fill_gradient(low="white", high="#009194") +
      labs(x = "Prediction",y = "Reference") +
      scale_y_discrete(labels=c("Dab","Idle","Running","Siu")) +
      scale_x_discrete(labels=c("Siu", "Running", "Idle", "Dab")) 
    
    
    ggplotly(rf_conf_mat)
  })
}


ui <- dashboardPage(
  dashboardHeader(title = "Model Dashboard"),
  ## Sidebar content
  dashboardSidebar(
    sidebarMenu(
      menuItem("EDA", tabName = "EDA", icon = icon("th")),
      menuItem("Model", tabName = "Model", icon = icon("th"))
    )
  ),
  dashboardBody( 
    ## Body content
    tabItems(
      
      # First tab content
      tabItem(tabName = "EDA",
              fluidRow(
                #First
                box(
                  title = "Collected Data"
                  ,status = "primary"
                  ,solidHeader = TRUE 
                  ,collapsible = TRUE ,
                  plotOutput("vol_plot", height = 500)),
                #Second
                box(
                  title = "Pairplot"
                  ,status = "primary"
                  ,solidHeader = TRUE 
                  ,collapsible = TRUE ,
                  plotOutput("CM_plot", height = 500))
                
                
                
              ),
              fluidRow(
                #Third
                box(
                  title = "Box Plot"
                  ,status = "primary"
                  ,solidHeader = TRUE 
                  ,collapsible = TRUE ,
                  plotOutput("state_plot", height = 500)),
                
                #Four
                box(
                  title = "Correlation Plot"
                  ,status = "primary"
                  ,solidHeader = TRUE 
                  ,collapsible = TRUE ,
                  plotOutput("pred_plot", height = 500))
                
              )

              
      ),
      
      # Second tab content
      tabItem(tabName = "Model",
              h2("Widgets tab content"),
              fluidRow(
                #Third
                box(
                  title = "Box Plot"
                  ,status = "primary"
                  ,solidHeader = TRUE 
                  ,collapsible = TRUE ,
                  plotOutput("Confusion_matrix_plot", height = 500)),
                
                #Four
                box(
                  title = "Correlation Plot"
                  ,status = "primary"
                  ,solidHeader = TRUE 
                  ,collapsible = TRUE ,
                  plotOutput("test2", height = 500))
                
              )
      )
    )
  )
)

shinyApp(ui, server)
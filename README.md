## **Data-Science-Project**
## **Group 6 Members**

---

### *Ma Estela Arenas*
### *Sean Howman*
### *Yuxiao Liu*
### *Feng Nie*

---

- Some data cleaning has been done on weather data.
- Data cleaning on Price table, VIC colume deleted, convert time into date format and then format into string format MM-dd,a,HH:mm. so that later on, we can easily grab the date and time. convert demand to INT, as the floating point number is not that important. No change to the pricing category, all data there are looking great.
- The reason I kept the AM/PM data is because there are separate temperature data on the weather table, and have yet decided its relatedness.
- Domain Knowledge *https://www.sciencedirect.com/science/article/pii/S014098832200189X*
• Temperature has robust and flat effects on electricity demand across all periods.
• Rain and sunshine have greater potential to affect people’s consumption behaviour.
• Sunshine sensitivity increases from late afternoon and peaks in early evening.
• More rain-sensitive activities occur before mid-afternoon during weekdays.
• In the mornings, sunlight has positive effects on weekend consumptions.
- As the above example shows, there is no point to keep any data which are NOT Rain, Sunshine or Temperature related. Please see the cleaned weather data.
-
-
-
# Exemple de code basique en R
a <- 5
b <- 3

somme <- a + b
produit <- a * b

print(paste("Somme =", somme))
print(paste("Produit =", produit))

notes <- c(12, 15, 9, 18, 13)

moyenne <- mean(notes)
mediane <- median(notes)
ecart_type <- sd(notes)

print(paste("Moyenne =", moyenne))
print(paste("Médiane =", mediane))
print(paste("Écart-type =", ecart_type))

plot(
  notes,
  type = "o",                # 'o' = points + lignes
  col = "blue",
  main = "Notes des étudiants",
  xlab = "Étudiant",
  ylab = "Note"
)

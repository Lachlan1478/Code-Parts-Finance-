
Function MoveToText()

Dim Ticker As String
Dim Sheetname As String

Sheetname = "ASX200"

Count = LstRec(Sheetname)

FilePath = "C:\Programming\ASX200_Tickers.txt"
Open FilePath For Output As #1

For i = 3 To Count
    Ticker = ThisWorkbook.Sheets(Sheetname).Cells(i, 1)
    Write #1, (Ticker)
    
Next i

Close #1
    

End Function

'count records
Function LstRec(Sheetname As String) As Long
Dim Count As Long
    Count = 0
    With Sheets(Sheetname)
GoAgain:
    Count = Count + 1
    If .Cells(Count, 1) <> "" Then GoTo GoAgain
    End With
    LstRec = Count - 2
    
    

End Function

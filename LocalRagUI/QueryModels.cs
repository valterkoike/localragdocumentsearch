namespace LocalRagUI.Models;

public class QueryRequest
{
    public string Question { get; set; } = string.Empty;
}

public class QueryResponse
{
    public string Question { get; set; } = string.Empty;
    public string Answer { get; set; } = string.Empty;
    public List<string> Sources { get; set; } = new();
}
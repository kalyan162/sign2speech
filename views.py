from subprocess import run, CalledProcessError, PIPE
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import sys


def index(request):
    return render(request, 'index.html')  # Ensure the path is correct


# def simple_function(request):
#     print("/nThis is a simple function")
#     return HttpResponse("""<html><h1>Hello</h1></html>""")

# def run_script(request):
#     try:
#             # Replace 'inference_classifier.py' with the path to your Python file
#             result = subprocess.run(['python', 'inference_classifier.py'], check=True, capture_output=True, text=True)
#             return JsonResponse({'status': 'success', 'message': 'Script executed successfully!', 'output': result.stdout})
#     except subprocess.CalledProcessError as e:
#             return JsonResponse({'status': 'error', 'message': str(e), 'output': e.stderr})
#     return JsonResponse({'status': 'error', 'message': 'Invalid request method.'})


def external(request):
    output = None
    error = None

    if request.method == "POST":
        # Extracting the parameter from the form submission
        param = request.POST.get('param')
        
        try:
            # Running the external script with the parameter
            result = run(
                [sys.executable, "C:/Users/HP/OneDrive/Desktop/finalpro/djangoreact/djangoproj/inference_classifier.py", param], 
                stdout=PIPE, stderr=PIPE, text=True, check=True
            )
            
            # Getting the script output and error
            output = result.stdout
            error = result.stderr
        except CalledProcessError as e:
            # Detailed error capturing for debugging
            error = f"Script failed with error code {e.returncode}\n"
            error += f"Command: {e.cmd}\n"
            error += f"Output: {e.output}\n"
            error += f"Error: {e.stderr}\n"
    
    # Passing the output, error, and the submitted parameter back to the template
    context = {
        'output': output,
        'error': error,
        'data_external': param if request.method == "POST" else ''
    }
    
    return render(request, "index.html", context)

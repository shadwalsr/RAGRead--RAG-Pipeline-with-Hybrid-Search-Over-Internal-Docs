import time
import re

# shared retry wrapper for all Gemini API calls
# handles the 429 RESOURCE_EXHAUSTED error by waiting and retrying

MAX_RETRIES = 3
DEFAULT_WAIT = 25  # seconds — slightly above Gemini's suggested 20s

def call_with_retry(fn, *args, retries=MAX_RETRIES, **kwargs):
    """
    Wraps any Gemini API call with automatic retry on rate-limit errors.
    
    Usage:
        result = call_with_retry(client.models.generate_content, model="...", contents="...")
    """
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err_str = str(e)
            
            # check if it's a rate limit error (429)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                # try to parse the suggested retry delay from the error
                wait_time = DEFAULT_WAIT
                delay_match = re.search(r'retryDelay.*?(\d+)', err_str)
                if delay_match:
                    wait_time = int(delay_match.group(1)) + 5  # add buffer
                
                if attempt < retries:
                    print(f"  [WAIT] rate limited (attempt {attempt}/{retries}), waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  [FAIL] rate limit: max retries ({retries}) exhausted")
                    raise
            else:
                # not a rate limit error — don't retry, just raise
                raise
